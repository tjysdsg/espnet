from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import copy

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator as ASRErrorCalculator
from espnet.nets.e2e_mt_common import ErrorCalculator as MTErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.nets_utils import pad_list
from sklearn.metrics import f1_score
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.gan_tts.abs_gan_tts import AbsGANTTS
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from itertools import groupby
from espnet2.layers.global_mvn import GlobalMVN
from espnet.nets.batch_beam_search import BatchBeamSearch, BatchHypothesis
from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.scorers.ctc import CTCPrefixScorer

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


def repeat_samples_in_batch(xs, length, fill=0, times=1):
    assert len(xs.shape) >= 2
    assert xs.size(0) == len(length)

    ret = xs.data.new(xs.size(0) * times, *xs.shape[1:]).fill_(fill)
    k = 0
    new_length = length.new(len(length) * times)
    for i, l in enumerate(length):
        for _ in range(times):
            ret[k, :l] = xs[i, :l]
            new_length[k] = length[i]
            k += 1
    return ret, new_length


class ESPnetGANTTSMDModel(AbsESPnetModel):
    """CTC-attention hybrid GANTTS model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        asr_encoder: AbsEncoder,
        asr_decoder: AbsDecoder,
        ctc: CTC,
        feats_extract: Optional[AbsFeatsExtract],
        pitch_extract: Optional[AbsFeatsExtract],
        energy_extract: Optional[AbsFeatsExtract],
        normalize: Optional[AbsNormalize and InversibleInterface],
        pitch_normalize: Optional[AbsNormalize and InversibleInterface],
        energy_normalize: Optional[AbsNormalize and InversibleInterface],
        tts: AbsGANTTS,
        asr_weight: float = 0.5,
        mt_weight: float = 0.0,
        mtlalpha: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        report_bleu: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
        speech_attn: bool = False,
        use_unpaired: bool = False,
        asr_normalize:bool = False,
        gumbel_softmax: bool = False,
        create_KL_copy: bool = False,
        teacher_student: bool = False,  # not used right now
        text_embed_loss_scale: float = 0.0,
        text_embed_loss: str = "mse",
        use_reinforce: bool = False,
        reinforce_sample_size: int = 4,
    ):
        assert check_argument_types()
        assert 0.0 <= asr_weight <= 1.0, "asr_weight should be [0.0, 1.0]"
        assert 0.0 <= mt_weight < 1.0, "mt_weight should be [0.0, 1.0)"
        assert 0.0 <= mtlalpha <= 1.0, "mtlalpha should be [0.0, 1.0]"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.feats_extract = feats_extract
        self.pitch_extract = pitch_extract
        self.energy_extract = energy_extract
        self.normalize = normalize
        self.pitch_normalize = pitch_normalize
        self.energy_normalize = energy_normalize
        self.tts = tts
        assert hasattr(
            tts, "generator"
        ), "generator module must be registered as tts.generator"
        assert hasattr(
            tts, "discriminator"
        ), "discriminator module must be registered as tts.discriminator"

        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.asr_weight = asr_weight
        self.mt_weight = mt_weight
        self.mtlalpha = mtlalpha
        self.token_list = token_list.copy()
        self.speech_attn = speech_attn
        self.specaug = None
        self.preencoder = None
        self.postencoder = None
        if asr_normalize==True:
            self.asr_normalize=GlobalMVN('/ocean/projects/cis210027p/jtang1/C/exp/backup_tts_stats_raw_char_360/train/feats_stats.npz')
            self.normalize = self.asr_normalize
        else:
            self.asr_normalize=None

        self.frontend = frontend
        self.asr_encoder = asr_encoder
        self.create_KL_copy = create_KL_copy
        self.teacher_student = teacher_student
        if self.create_KL_copy:
            self.asr_encoder_copy = copy.deepcopy(asr_encoder)
        self.asr_decoder = asr_decoder

        self.use_unpaired = use_unpaired
        self.idx_blank = self.token_list.index(sym_blank)
        self.idx_space = self.token_list.index(sym_space)

        # decoder's output -> gumbel softmax -> TTS text encoder -> TTS encoder
        self.gumbel_softmax = gumbel_softmax
        if self.gumbel_softmax:
            assert not self.tts.skip_text_encoder
            assert self.tts.gumbel_softmax_input
        elif self.use_unpaired:
            # decoder's hidden states -> TTS encoder
            assert self.tts.skip_text_encoder or use_reinforce

        self.criterion_asr = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # submodule for ASR task
        assert asr_decoder is not None, "ASR decoder needs to be present for MD"
        # assert (
        #     src_token_list is not None
        # ), "Missing src_token_list, cannot add asr module to st model"
        if self.mtlalpha > 0.0:
            self.ctc = ctc
        # import pdb;pdb.set_trace()
        self.asr_decoder = asr_decoder
        self.asr_decoder.gumbel_softmax = self.gumbel_softmax

        # MT error calculator
        if report_bleu:
            self.mt_error_calculator = MTErrorCalculator(
                token_list, sym_space, sym_blank, report_bleu
            )
        else:
            self.mt_error_calculator = None

        # ASR error calculator
        if report_cer or report_wer:
            self.asr_error_calculator = ASRErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.asr_error_calculator = None

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.feats_dim = 256
        self.linear_layer_y_pred = torch.nn.Linear(in_features=self.feats_dim, out_features=self.feats_dim).to("cuda:0") # fixme: hardcoding device for now.

        self.text_embed_loss_scale = text_embed_loss_scale
        self.text_embed_loss_method = text_embed_loss
        assert self.text_embed_loss_method == 'mse' or self.text_embed_loss_method == 'kl'

        # ASR beam search if REINFORCE is enabled
        self.use_reinforce = use_reinforce
        self.reinforce_sample_size = reinforce_sample_size
        if use_reinforce:
            scorers = dict(
                decoder=self.asr_decoder,
                # ctc=CTCPrefixScorer(ctc=self.ctc, eos=self.eos),  # TODO: use CTC in beam search?
            )

            weights = dict(
                decoder=1.0 - self.mtlalpha,
                # ctc=self.mtlalpha,
            )
            token_list = self.token_list
            self.beam_search = BatchBeamSearch(
                beam_size=10,  # FIXME: hard-coded
                weights=weights,
                scorers=scorers,
                sos=self.sos,
                eos=self.eos,
                vocab_size=len(token_list),
                token_list=token_list,
                pre_beam_score_key=None if self.mtlalpha == 1.0 else "full",
            )

            # beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
            # for scorer in scorers.values():
            #     if isinstance(scorer, torch.nn.Module):
            #         scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
            # logging.info(f"Decoding device={device}, dtype={dtype}")

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        sudo_text: Optional[torch.Tensor] = None,
        sudo_text_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        forward_generator: bool=True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
            text: (Batch, Length)
            text_lengths: (Batch,)
            sudo_text: Pseudo-labels used for unpaired training, will use pretrained encoder ctc output if none
            sudo_text_lengths: Pseudo-labels used for unpaired training
            spembs:
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        # print(text_lengths.shape)
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        # additional checks with valid src_text
        if sudo_text is not None:
            assert sudo_text_lengths.dim() == 1, sudo_text_lengths.shape
            assert text.shape[0] == sudo_text.shape[0] == sudo_text_lengths.shape[0], (
                text.shape,
                sudo_text.shape,
                sudo_text_lengths.shape,
            )

        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]
        if sudo_text is not None:
            sudo_text = sudo_text[:, : sudo_text_lengths.max()]

        # 1. Encoder
        if self.create_KL_copy:
            encoder_out, encoder_out_lens, mse_loss = self.encode(
                speech, speech_lengths
            )
        else:
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # 2a. Pseudo-labels
        # If not given:
        #     - use canonical text during validation for calculating loss
        #     - use reference encoder's output during training
        if self.use_unpaired and sudo_text is None:
            if self.training:
                assert False  # assume we always have sudo text during unpaired training
                # sudo_text = y_ctc_gold
                # sudo_text_lengths = y_ctc_gold_lens
            else:
                sudo_text = text
                sudo_text_lengths = text_lengths

        # 2b. CTC branch
        if self.use_unpaired:
            (
                y_ctc_pred_pad,
                seq_hat_total_lens,
                loss_asr_ctc,
                cer_asr_ctc,
            ) = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, sudo_text, sudo_text_lengths, ground_truth=text,
            )
        elif self.mtlalpha > 0:
            loss_asr_ctc, cer_asr_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
        else:
            loss_asr_ctc, cer_asr_ctc = 0, None

        if self.use_unpaired:
            dec_asr_lengths = sudo_text_lengths + 1
        else:
            dec_asr_lengths = text_lengths + 1

        # 2c. ASR Decoder
        if self.use_unpaired:
            (
                loss_asr_att,
                acc_asr_att,
                cer_asr_att,
                wer_asr_att,
                y_pred,
            ) = self._calc_asr_att_loss(
                encoder_out, encoder_out_lens, sudo_text, sudo_text_lengths, ground_truth=text,
            )
            if self.gumbel_softmax:
                dec_asr_lengths = dec_asr_lengths.to(dtype=int)
        else:
            (
                loss_asr_att,
                acc_asr_att,
                cer_asr_att,
                wer_asr_att,
                y_pred,
            ) = self._calc_asr_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 2d. ASR losses
        loss_asr = self.mtlalpha * loss_asr_ctc + (1 - self.mtlalpha) * loss_asr_att

        # MSE loss between ASR decoder output embeddings and TTS encoder text embeddings
        text_embed_loss = self.calc_text_embed_loss(
            y_pred,
            sudo_text if self.use_unpaired else text,
            sudo_text_lengths if self.use_unpaired else text_lengths,
        )

        # 2e. Prepare TTS features
        with autocast(False):
            # Extract features
            if self.feats_extract is not None:
                feats, feats_lengths = self.feats_extract(speech, speech_lengths)
            else:
                # Use precalculated feats (feats_type != raw case)
                feats, feats_lengths = speech, speech_lengths

            # Extract auxiliary features
            # if self.pitch_extract is not None and pitch is None:
            #     pitch, pitch_lengths = self.pitch_extract(
            #         speech,
            #         speech_lengths,
            #         feats_lengths=feats_lengths,
            #         durations=durations,
            #         durations_lengths=durations_lengths,
            #     )
            # if self.energy_extract is not None and energy is None:
            #     energy, energy_lengths = self.energy_extract(
            #         speech,
            #         speech_lengths,
            #         feats_lengths=feats_lengths,
            #         durations=durations,
            #         durations_lengths=durations_lengths,
            #     )

            # Normalize
            # FIXME: adhoc solution
            # self.normalize = None
            # FIXME: commented out the below 2 lines for now (normalization size mismatch)
            # if self.normalize is not None:
            #     feats, feats_lengths = self.normalize(feats, feats_lengths)
            # if self.pitch_normalize is not None:
            #     pitch, pitch_lengths = self.pitch_normalize(pitch, pitch_lengths)
            # if self.energy_normalize is not None:
            #     energy, energy_lengths = self.energy_normalize(energy, energy_lengths)

        # 2e. TTS
        batch = dict(
            text=y_pred,
            text_lengths=dec_asr_lengths,
            forward_generator=forward_generator,
        )
        if feats is not None:
            batch.update(feats=feats, feats_lengths=feats_lengths)
        if self.speech_attn:
            batch.update(speech_embed=encoder_out)
            batch.update(speech_embed_lengths=encoder_out_lens)
            assert False  # FIXME: remove speech_attn code
        if spembs is not None:
            batch.update(spembs=spembs)
        batch.update(speech=speech, speech_lengths=speech_lengths)

        # 2f. TTS
        if self.use_reinforce:  # REINFORCE, see also https://github.com/creatorscan/espnet-asrtts
            assert forward_generator

            if self.training:
                decoder_out = y_pred
                # Repeat text hypotheses and perform weighted random sampling
                decoder_out, dec_asr_lengths = repeat_samples_in_batch(
                    decoder_out, dec_asr_lengths, fill=0, times=self.reinforce_sample_size
                )
                # text's last dim is softmax probs here, see self._calc_asr_att_loss
                max_seq_len = decoder_out.shape[1]
                sampled_text = []
                tts_text = F.softmax(decoder_out, dim=-1)
                for i, seq in enumerate(tts_text):
                    token_ids = [
                        torch.multinomial(seq[j], num_samples=1).item() if torch.sum(seq[j]) > 0 else 0
                        for j in range(max_seq_len)
                    ]
                    sampled_text.append(token_ids)
                sampled_text = torch.as_tensor(sampled_text, dtype=torch.long, device=tts_text.device)

                # update TTS input batch
                speech, speech_lengths = repeat_samples_in_batch(
                    speech, speech_lengths, fill=0, times=self.reinforce_sample_size
                )
                if feats is not None:
                    feats, feats_lengths = repeat_samples_in_batch(
                        feats, feats_lengths, fill=0, times=self.reinforce_sample_size
                    )
                if spembs is not None:
                    spembs = torch.repeat_interleave(spembs, self.reinforce_sample_size, dim=0)

                batch_size = sampled_text.shape[0]
                decoder_out = F.log_softmax(decoder_out, dim=-1)
                log_probs = torch.stack([
                    F.nll_loss(decoder_out[i], sampled_text[i], reduction='sum') for i in range(batch_size)
                ])

                rewards = []
                for i in range(batch_size):
                    batch.update(text=sampled_text[[i], :dec_asr_lengths[i]], text_lengths=dec_asr_lengths[[i]])
                    batch.update(speech=speech[[i], :speech_lengths[i]], speech_lengths=speech_lengths[[i]])
                    batch.update(feats=feats[[i], :feats_lengths[i]], feats_lengths=feats_lengths[[i]])
                    batch.update(spembs=spembs[[i]])
                    with torch.no_grad():
                        vits_dict = self.tts.forward_reinforce(**batch)
                        # vits_dict = self.tts(**batch, full=True)
                        rewards.append(-vits_dict['loss'].item())

                rewards = torch.as_tensor(rewards, device=log_probs.device)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

                loss = log_probs * rewards
                loss = loss.mean()
                # self.asr_decoder.decoders[0].feed_forward.w_1.weight.grad
                # self.asr_encoder.encoders[0].feed_forward.w_1.weight.grad
            else:
                loss = loss_asr
                vits_dict = {}

            stats = dict(
                loss=loss,
                loss_asr=loss_asr,
                acc_asr=acc_asr_att,
                cer_ctc=cer_asr_ctc,
                cer=cer_asr_att,
                wer=wer_asr_att,
            )
            # force_gatherable: to-device and to-tensor if scalar for DataParallel
            gathered_loss, gathered_stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

            vits_dict['loss'] = gathered_loss
            vits_dict['stats'] = gathered_stats
            vits_dict['weight'] = weight
            return vits_dict

        else:  # Normal TTS
            vits_dict = self.tts(**batch, full=True)
            tts_loss = vits_dict['loss']
            tts_stats = vits_dict['stats']
            # tts_weight = vits_dict['weight']
            is_discriminator = vits_dict['optim_idx']

            loss = (1 - self.asr_weight) * tts_loss + self.asr_weight * loss_asr + text_embed_loss

            if not is_discriminator:  # generator
                stats = dict(
                    loss=loss.detach(),
                    loss_asr=loss_asr.detach() if type(loss_asr) is not float else loss_asr,
                    acc_asr=acc_asr_att,
                    cer_ctc=cer_asr_ctc,
                    cer=cer_asr_att,
                    wer=wer_asr_att,
                    text_embed_loss=text_embed_loss.detach() if type(text_embed_loss) is not float else text_embed_loss,
                    tts_generator_adv_loss=tts_stats["generator_adv_loss"],
                    tts_generator_dur_loss=tts_stats["generator_dur_loss"],
                    tts_generator_feat_match_loss=tts_stats["generator_feat_match_loss"],
                    tts_generator_kl_loss=tts_stats["generator_kl_loss"],
                    tts_generator_loss=tts_stats["generator_loss"],
                    tts_generator_mel_loss=tts_stats["generator_mel_loss"],
                )
                # force_gatherable: to-device and to-tensor if scalar for DataParallel
                gathered_loss, gathered_stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

                vits_dict['loss'] = gathered_loss
                vits_dict['stats'] = gathered_stats
                vits_dict['weight'] = weight

                return vits_dict

            else:  # discriminator
                # import pdb;pdb.set_trace()
                stats = dict(
                    # tts
                    tts_discriminator_loss=tts_stats.get("discriminator_loss", 0),
                    tts_discriminator_fake_loss=tts_stats.get("discriminator_fake_loss", 0),
                    tts_discriminator_real_loss=tts_stats.get("discriminator_real_loss", 0),
                )
                # force_gatherable: to-device and to-tensor if scalar for DataParallel
                gathered_loss, gathered_stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

                # vits_dict['loss'] = gathered_loss
                vits_dict['stats'] = gathered_stats
                vits_dict['weight'] = weight

                return vits_dict

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        sudo_text: Optional[torch.Tensor] = None,
        sudo_text_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by st_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)
            # if self.asr_normalize is not None:
            #     feats, feats_lengths = self.asr_normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.asr_encoder(feats, feats_lengths)

        if self.create_KL_copy:
            # import pdb;pdb.set_trace()
            mse_criterion = torch.nn.MSELoss(reduction="mean")
            encoder_out_copy, encoder_out_copy_lens, _ = self.asr_encoder_copy(
                feats, feats_lengths
            )
            mse_loss = mse_criterion(encoder_out, encoder_out_copy)

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        if self.create_KL_copy:
            return encoder_out, encoder_out_lens, mse_loss
        else:
            return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def calc_text_embed_loss(self, x: torch.Tensor, text: torch.Tensor, text_lengths: torch.Tensor):
        if self.text_embed_loss_scale == 0:
            # return 0
            return torch.tensor(0.0).to(x.device)

        # apply linear layer
        x = self.linear_layer_y_pred(x)

        tgt, pad_mask = self.tts.generator.text_encoder.encode(text, text_lengths)
        tgt.masked_fill_(pad_mask, 0.0)

        x = x[:, :-1, :].masked_fill(pad_mask, 0.0)

        if self.text_embed_loss_method == "mse":
            loss = F.mse_loss(x, tgt)
        else:
            loss = F.kl_div(F.log_softmax(x, dim=-1), F.softmax(tgt, dim=-1), reduction="batchmean")

        return self.text_embed_loss_scale * loss

    def _calc_asr_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        ground_truth: torch.Tensor = None,  # only used to compute CER, fallback to ys_pad if None
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(
            ys_pad, self.sos, self.eos, self.ignore_id
        )
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _, hs_dec_asr = self.asr_decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens, return_hidden=True
        )

        # 2. Compute attention loss
        loss_att = self.criterion_asr(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        cer_att, wer_att = None, None
        if not self.training and self.asr_error_calculator is not None:
            ys_hat = decoder_out.argmax(dim=-1)

            if ground_truth is None:
                ground_truth = ys_pad
            cer_att, wer_att = self.asr_error_calculator(ys_hat.cpu(), ground_truth.cpu())

        if self.gumbel_softmax:
            y_pred_gumbel = F.gumbel_softmax(decoder_out, tau=1.0, hard=True, dim=-1)
            return loss_att, acc_att, cer_att, wer_att, y_pred_gumbel
        elif self.use_reinforce:
            return loss_att, acc_att, cer_att, wer_att, decoder_out
        else:
            return loss_att, acc_att, cer_att, wer_att, hs_dec_asr

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        ground_truth: torch.Tensor = None,  # only used to compute CER, fallback to ys_pad if None
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc ys_hat
        ys_hat = self.ctc.argmax(encoder_out).data

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.asr_error_calculator is not None:
            if ground_truth is None:
                ground_truth = ys_pad
            cer_ctc = self.asr_error_calculator(ys_hat.cpu(), ground_truth.cpu(), is_ctc=True)

        if self.use_unpaired:
            seq_hat_total = []
            # import pdb;pdb.set_trace()
            for i, y in enumerate(ys_hat):
                y_hat = [x[0] for x in groupby(y)]
                seq_hat = []
                for idx in y_hat:
                    idx = int(idx)
                    if idx != -1 and idx != self.idx_blank:
                        seq_hat.append(idx)
                seq_hat_total.append(
                    torch.Tensor(seq_hat).to(ys_hat.device, dtype=ys_hat.dtype)
                )
            # import pdb;pdb.set_trace()
            seq_hat_total_lens = torch.Tensor([len(k) for k in seq_hat_total]).to(
                ys_hat.device, dtype=ys_hat.dtype
            )
            y_ctc_pred_pad = pad_list(seq_hat_total, self.ignore_id)
            return y_ctc_pred_pad, seq_hat_total_lens, loss_ctc, cer_ctc

        return loss_ctc, cer_ctc

    def _calc_ctc_output(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ):
        ys_hat = self.ctc_copy.argmax(encoder_out).data
        seq_hat_total = []
        # import pdb;pdb.set_trace()
        for i, y in enumerate(ys_hat):
            y_hat = [x[0] for x in groupby(y)]
            seq_hat = []
            for idx in y_hat:
                idx1 = int(idx)
                if idx1 != -1 and idx1 != self.idx_blank:
                    seq_hat.append(idx)
            seq_hat_total.append(
                torch.Tensor(seq_hat).to(ys_hat.device, dtype=ys_hat.dtype)
            )
        # import pdb;pdb.set_trace()
        seq_hat_total_lens = torch.Tensor([len(k) for k in seq_hat_total]).to(
            ys_hat.device, dtype=ys_hat.dtype
        )
        y_ctc_pred_pad = pad_list(seq_hat_total, self.ignore_id)
        return y_ctc_pred_pad, seq_hat_total_lens

    def tts_inference(
        self,
        text: torch.Tensor,
        speech: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        **decode_config,
    ) -> Dict[str, torch.Tensor]:
        """Caclualte features and return them as a dict.

        Args:
            text (Tensor): Text index tensor (T_text).
            speech (Tensor): Speech waveform tensor (T_wav).
            spembs (Optional[Tensor]): Speaker embedding tensor (D,).
            sids (Optional[Tensor]): Speaker ID tensor (1,).
            lids (Optional[Tensor]): Language ID tensor (1,).
            durations (Optional[Tensor): Duration tensor.
            pitch (Optional[Tensor): Pitch tensor.
            energy (Optional[Tensor): Energy tensor.

        Returns:
            Dict[str, Tensor]: Dict of outputs.

        """
        input_dict = dict(text=text)

        if spembs is not None:
            input_dict.update(spembs=spembs)
        if sids is not None:
            input_dict.update(sids=sids)
        if lids is not None:
            input_dict.update(lids=lids)

        output_dict = self.tts.inference(**input_dict, **decode_config)

        if self.normalize is not None and output_dict.get("feat_gen") is not None:
            # NOTE: normalize.inverse is in-place operation
            feat_gen_denorm = self.normalize.inverse(
                output_dict["feat_gen"].clone()[None]
            )[0][0]
            output_dict.update(feat_gen_denorm=feat_gen_denorm)

        return output_dict
