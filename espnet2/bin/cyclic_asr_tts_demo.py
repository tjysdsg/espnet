#!/usr/bin/env python3
import logging
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union
import torch
from typeguard import check_argument_types
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.beam_search import BeamSearch
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet2.tasks.lm import LMTask
from espnet2.tasks.gan_tts import GANTTSTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter


class CyclicASRTTS:
    def __init__(
            self,
            train_config: Union[Path, str] = None,
            model_file: Union[Path, str] = None,
            lm_train_config: Union[Path, str] = None,
            lm_file: Union[Path, str] = None,
            ngram_scorer: str = "full",
            ngram_file: Union[Path, str] = None,
            token_type: str = None,
            bpemodel: str = None,
            device: str = "cpu",
            maxlenratio: float = 0.0,
            minlenratio: float = 0.0,
            batch_size: int = 1,
            dtype: str = "float32",
            beam_size: int = 20,
            ctc_weight: float = 0.5,
            lm_weight: float = 1.0,
            ngram_weight: float = 0.9,
            penalty: float = 0.0,
            nbest: int = 1,
    ):
        assert check_argument_types()

        # 1. Build ASR model
        scorers = {}
        cyclic_model, cyclic_train_args = GANTTSTask.build_model_from_file(
            train_config, model_file, device
        )
        cyclic_model.to(dtype=getattr(torch, dtype)).eval()

        asr_decoder = cyclic_model.asr_decoder

        ctc = CTCPrefixScorer(ctc=cyclic_model.ctc, eos=cyclic_model.eos)
        token_list = cyclic_model.token_list
        scorers.update(
            decoder=asr_decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
        )

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            scorers["lm"] = lm.lm

        # 3. Build ngram model
        if ngram_file is not None:
            if ngram_scorer == "full":
                from espnet.nets.scorers.ngram import NgramFullScorer

                ngram = NgramFullScorer(ngram_file, token_list)
            else:
                from espnet.nets.scorers.ngram import NgramPartScorer

                ngram = NgramPartScorer(ngram_file, token_list)
        else:
            ngram = None
        scorers["ngram"] = ngram

        # 4. Build BeamSearch object
        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            ngram=ngram_weight,
            length_bonus=penalty,
        )
        beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=cyclic_model.sos,
            eos=cyclic_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        )

        # TODO(karita): make all scorers batchfied
        if batch_size == 1:
            non_batch = [
                k
                for k, v in beam_search.full_scorers.items()
                if not isinstance(v, BatchScorerInterface)
            ]
            if len(non_batch) == 0:
                beam_search.__class__ = BatchBeamSearch
                logging.info("BatchBeamSearch implementation is selected.")
            else:
                logging.warning(
                    f"As non-batch scorers {non_batch} are found, "
                    f"fall back to non-batch implementation."
                )

        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
        logging.info(f"Beam_search: {beam_search}")
        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 5. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = cyclic_train_args.token_type
        if bpemodel is None:
            bpemodel = cyclic_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.cyclic_model = cyclic_model
        self.cyclic_train_args = cyclic_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest

    @torch.no_grad()
    def __call__(
            self,
            speech: torch.Tensor,
            text: str,
            sudo_text: str,
            spembs: torch.Tensor,
            reinforce=False,
            reinforce_sample_size: int = 4,
    ):
        speech_lengths = torch.as_tensor([speech.shape[1]], dtype=torch.long, device=speech.device)

        text = self.converter.tokens2ids(self.tokenizer.text2tokens(text))
        text = torch.as_tensor([text], dtype=torch.long)
        text_lengths = torch.as_tensor([text.shape[1]], dtype=torch.long, device=text.device)

        sudo_text = self.converter.tokens2ids(self.tokenizer.text2tokens(sudo_text))
        sudo_text = torch.as_tensor([sudo_text], dtype=torch.long)
        sudo_text_lengths = torch.as_tensor([sudo_text.shape[1]], dtype=torch.long, device=text.device)

        return self.cyclic_model(
            speech=speech,
            speech_lengths=speech_lengths,
            text=text,
            text_lengths=text_lengths,
            sudo_text=sudo_text,
            sudo_text_lengths=sudo_text_lengths,
            spembs=spembs,
            reinforce=reinforce,
            reinforce_sample_size=reinforce_sample_size,
        )

    @staticmethod
    def from_pretrained(
            model_tag: Optional[str] = None,
            **kwargs: Optional[Any],
    ):
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return CyclicASRTTS(**kwargs)
