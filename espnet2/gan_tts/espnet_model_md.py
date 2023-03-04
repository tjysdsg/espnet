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

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


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
        intermediate_supervision: bool = False,
        teacher_student: bool = False,  # not used right now
        use_asr_decoder_loss: bool = True,
    ):
        assert check_argument_types()
        assert 0.0 <= asr_weight < 1.0, "asr_weight should be (0.0, 1.0)"
        assert 0.0 <= mt_weight < 1.0, "mt_weight should be [0.0, 1.0)"
        assert 0.0 <= mtlalpha < 1.0, "mtlalpha should be [0.0, 1.0)"

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
        self.intermediate_supervision = intermediate_supervision
        self.teacher_student = teacher_student
        if self.create_KL_copy or self.intermediate_supervision:
            self.asr_encoder_copy = copy.deepcopy(asr_encoder)
        self.asr_decoder = asr_decoder

        self.use_unpaired = use_unpaired
        self.use_asr_decoder_loss = use_asr_decoder_loss
        self.gumbel_softmax = False
        if self.use_unpaired:
            self.idx_blank = self.token_list.index(sym_blank)
            self.idx_space = self.token_list.index(sym_space)
            self.gumbel_softmax = gumbel_softmax

        # decoder's output -> gumbel softmax -> TTS text encoder -> TTS encoder
        if self.gumbel_softmax:
            assert not self.tts.skip_text_encoder
            assert self.tts.gumbel_softmax_input
        # elif self.use_unpaired:
        #     # decoder's hidden states -> TTS encoder
        #     assert self.tts.skip_text_encoder

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
            if self.intermediate_supervision:
                self.ctc_copy = copy.deepcopy(self.ctc)
        # import pdb;pdb.set_trace()
        self.asr_decoder = asr_decoder
        # self.asr_decoder.gumbel_softmax = self.gumbel_softmax

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
        # if self.intermediate_supervision:
        #     # y_ctc_gold is the CTC output of the pretrained encoder self.asr_encoder_copy
        #     y_ctc_gold, y_ctc_gold_lens, encoder_out, encoder_out_lens = self.encode(
        #         speech, speech_lengths
        #     )
        # elif self.create_KL_copy:
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
        if self.intermediate_supervision and sudo_text is None:
            if self.training:
                assert False  # assume we always have sudo text during unpaired training
                # sudo_text = y_ctc_gold
                # sudo_text_lengths = y_ctc_gold_lens
            else:
                sudo_text = text
                sudo_text_lengths = text_lengths

        # 2b. CTC branch
        if self.use_unpaired:
            if self.intermediate_supervision:
                (
                    y_ctc_pred_pad,
                    seq_hat_total_lens,
                    loss_asr_ctc,
                    cer_asr_ctc,
                ) = self._calc_ctc_loss(
                    encoder_out, encoder_out_lens, sudo_text, sudo_text_lengths, ground_truth=text,
                )
            else:
                (
                    y_ctc_pred_pad,
                    seq_hat_total_lens,
                    loss_asr_ctc,
                    cer_asr_ctc,
                ) = self._calc_ctc_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )
        elif self.mtlalpha > 0:
            loss_asr_ctc, cer_asr_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
        else:
            loss_asr_ctc, cer_asr_ctc = 0, None

        if self.use_unpaired:
            # if self.gumbel_softmax:
            #     dec_asr_lengths = seq_hat_total_lens + 1
            # else:
            dec_asr_lengths = sudo_text_lengths + 1
        else:
            dec_asr_lengths = text_lengths + 1

        # 2a. ASR Decoder
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
            # if self.gumbel_softmax:
            #     dec_asr_lengths = dec_asr_lengths.to(dtype=int)
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
        if spembs is not None:
            batch.update(spembs=spembs)
        batch.update(speech=speech, speech_lengths=speech_lengths)

        # import pdb; pdb.set_trace()
        # FIXME: adapt GANTTS api
        vits_dict = self.tts(**batch)
        tts_loss = vits_dict['loss']
        tts_stats = vits_dict['stats']
        tts_weight = vits_dict['weight']
        is_discriminator = vits_dict['optim_idx']
        # tts_loss, tts_stats, tts_weight = self.tts(**batch)

        # 3. Loss computation
        asr_ctc_weight = self.mtlalpha
        if asr_ctc_weight == 0.0:
            loss_asr = loss_asr_att
        elif self.use_asr_decoder_loss:
            loss_asr = (
                asr_ctc_weight * loss_asr_ctc + (1 - asr_ctc_weight) * loss_asr_att
            )
        else:
            loss_asr = loss_asr_ctc
        loss = (1 - self.asr_weight) * tts_loss + self.asr_weight * loss_asr
        # import pdb;pdb.set_trace()
        if self.create_KL_copy:
            # import pdb;pdb.set_trace()
            loss = 0.95 * loss + 0.05 * mse_loss
        # vits_dict['loss'] = loss
        # import pdb;pdb.set_trace()
        # print(tts_stats)
        # FIXME: figure out the output of self.tts(**batch), replace below
        if self.create_KL_copy:
            assert False
        elif self.intermediate_supervision:
            stats = dict(
                loss=loss.detach(),
                loss_asr=loss_asr.detach() if type(loss_asr) is not float else loss_asr,
                acc_asr=acc_asr_att,
                cer_ctc=cer_asr_ctc,
                cer=cer_asr_att,
                wer=wer_asr_att,
                # tts
                tts_loss=tts_loss.detach(),
                tts_discriminator_loss=tts_stats["discriminator_loss"],
                tts_discriminator_fake_loss=tts_stats["discriminator_fake_loss"],
                tts_discriminator_real_loss=tts_stats["discriminator_real_loss"],
                tts_generator_adv_loss=tts_stats["generator_adv_loss"],
                tts_generator_dur_loss=tts_stats["generator_dur_loss"],
                tts_generator_feat_match_loss=tts_stats["generator_feat_match_loss"],
                tts_generator_kl_loss=tts_stats["generator_kl_loss"],
                tts_generator_loss=tts_stats["generator_loss"],
                tts_generator_mel_loss=tts_stats["generator_mel_loss"],
            )

        if not is_discriminator:  # generator
            stats = dict(
                # loss = loss.detach(),
                # asr 
                loss_asr=loss_asr.detach() if type(loss_asr) is not float else loss_asr,
                acc_asr=acc_asr_att,
                cer_ctc=cer_asr_ctc,
                cer=cer_asr_att,
                wer=wer_asr_att,
                tts_generator_adv_loss=tts_stats["generator_adv_loss"],
                tts_generator_dur_loss=tts_stats["generator_dur_loss"],
                tts_generator_feat_match_loss=tts_stats["generator_feat_match_loss"],
                tts_generator_kl_loss=tts_stats["generator_kl_loss"],
                tts_generator_loss=tts_stats["generator_loss"],
                tts_generator_mel_loss=tts_stats["generator_mel_loss"],
            )
            # FIXME, TODO [DONE]: update vits_dict with new loss and stats
            # force_gatherable: to-device and to-tensor if scalar for DataParallel
            gathered_loss, gathered_stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

            vits_dict['loss'] = gathered_loss
            vits_dict['stats'] = gathered_stats
            vits_dict['weight'] = weight

            return vits_dict

        else:  # discriminator
            # import pdb;pdb.set_trace()
            stats = dict(
                # loss is independently registered, so ignore in stats
                # loss = loss.detach(),
                # # asr 
                # loss_asr=loss_asr.detach() if type(loss_asr) is not float else loss_asr,
                # acc_asr=acc_asr_att,
                # cer_ctc=cer_asr_ctc,
                # cer=cer_asr_att,
                # wer=wer_asr_att,
                # tts
                tts_loss=tts_loss.detach(),
                tts_discriminator_loss=tts_stats.get("discriminator_loss", 0),
                tts_discriminator_fake_loss = tts_stats.get("discriminator_fake_loss", 0),
                tts_discriminator_real_loss = tts_stats.get("discriminator_real_loss", 0),
            )
            # FIXME, TODO [DONE]: update vits_dict with new loss and stats
            # force_gatherable: to-device and to-tensor if scalar for DataParallel
            gathered_loss, gathered_stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

            # vits_dict['loss'] = gathered_loss
            vits_dict['stats'] = gathered_stats
            vits_dict['weight'] = weight

            return vits_dict

        # # FIXME, TODO [DONE]: update vits_dict with new loss and stats
        # # force_gatherable: to-device and to-tensor if scalar for DataParallel
        # gathered_loss, gathered_stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # vits_dict['loss'] = gathered_loss;
        # vits_dict['stats'] = gathered_stats;
        # vits_dict['weight'] = weight;

        # return vits_dict

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

        if self.intermediate_supervision:
            pass
            # encoder_out_copy, encoder_out_copy_lens, _ = self.asr_encoder_copy(
            #     feats, feats_lengths
            # )
            # y_ctc_gold, y_ctc_gold_lens = self._calc_ctc_output(
            #     encoder_out_copy, encoder_out_copy_lens
            # )
            # import pdb;pdb.set_trace()
        elif self.create_KL_copy:
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

        # if self.intermediate_supervision:
        #     return y_ctc_gold, y_ctc_gold_lens, encoder_out, encoder_out_lens
        # elif self.create_KL_copy:
        #     return encoder_out, encoder_out_lens, mse_loss
        # else:
        #     return encoder_out, encoder_out_lens
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

    def _calc_asr_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        ground_truth: torch.Tensor = None,  # only used to compute CER, fallback to ys_pad if None
    ):
        # if self.gumbel_softmax:
        #     # ys_pad is gumbel softmax of the encoder output
        #     gumbel_idx = torch.arange(ys_pad.shape[-1]).to(
        #         ys_pad.device, dtype=ys_pad.dtype
        #     )
        #     ys_gumbel = torch.matmul(ys_pad, gumbel_idx).to(dtype=int)
        #     _, ys_out_pad = add_sos_eos(ys_gumbel, self.sos, self.eos, self.ignore_id)
        #     ys_in_pad = torch.zeros(ys_pad.shape[-1], device=ys_pad.device, dtype=ys_pad.dtype)
        #     ys_in_pad[self.sos] = 1
        #     ys_in_pad = ys_in_pad.unsqueeze(0)
        #     # import pdb;pdb.set_trace()
        #     ys_in_pad = torch.stack([torch.cat([ys_in_pad, y], dim=0) for y in ys_pad])
        # else:
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
            y_pred_gumbel = F.gumbel_softmax(decoder_out, tau=1, hard=True, dim=-1)
            return loss_att, acc_att, cer_att, wer_att, y_pred_gumbel
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
        # if self.use_unpaired and self.gumbel_softmax:
        #     ys_one_hot_hat, ys_hat = self.ctc.gumbel_softmax(encoder_out)
        ys_hat = self.ctc.argmax(encoder_out).data

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.asr_error_calculator is not None:
            if ground_truth is None:
                ground_truth = ys_pad
            cer_ctc = self.asr_error_calculator(ys_hat.cpu(), ground_truth.cpu(), is_ctc=True)

        if self.use_unpaired:
            # if self.gumbel_softmax:
            #     seq_hat_total = []
            #     for i, y in enumerate(ys_hat):
            #         y_hat = [x[0] for x in groupby(y)]
            #         y_hat_sum = [sum(1 for _ in x[1]) for x in groupby(y)]
            #         seq_hat = []
            #         idx_index = 0
            #         # import pdb;pdb.set_trace()
            #         for idx_len in range(len(y_hat)):
            #             idx = int(y_hat[idx_len])
            #             if idx != -1 and idx != self.idx_blank:
            #                 seq_hat.append(ys_one_hot_hat[i][idx_index])
            #             idx_index += y_hat_sum[idx_len]
            #         # import pdb;pdb.set_trace()
            #         seq_hat_total.append(
            #             torch.stack(seq_hat).to(ys_hat.device, dtype=ys_hat.dtype)
            #         )
            #     seq_hat_total_lens = torch.Tensor([len(k) for k in seq_hat_total]).to(
            #         ys_hat.device, dtype=ys_hat.dtype
            #     )
            #     n_batch = len(seq_hat_total)
            #     max_len = max(x.size(0) for x in seq_hat_total)
            #     xs = seq_hat_total
            #     pad = xs[0].new_zeros(n_batch, max_len, *xs[0].size()[1:])
            #     pad[:, :, 0] = 1.0
            #     for i in range(n_batch):
            #         pad[i, : xs[i].size(0)] = xs[i]
            #     # import pdb;pdb.set_trace()
            #     y_ctc_pred_pad = pad
            # else:
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
