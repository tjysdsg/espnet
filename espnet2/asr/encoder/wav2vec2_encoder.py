# Copyright 2021 Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
import contextlib
import copy
import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from filelock import FileLock
from typeguard import check_argument_types
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq import utils

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class FairSeqWav2Vec2Encoder(AbsEncoder):
    """FairSeq Wav2Vec2 encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        w2v_url: url to Wav2Vec2.0 pretrained model
        w2v_dir_path: directory to download the Wav2Vec2.0 pretrained model.
        normalize_before: whether to use layer_norm before the first block
        finetune_last_n_layers: last n layers to be finetuned in Wav2Vec2.0
                                0 means to finetune every layer if freeze_w2v=False.
    """

    def __init__(
        self,
        input_size: int,
        w2v_url: str,
        w2v_dir_path: str = "./",
        output_size: int = 256,
        normalize_before: bool = False,
        freeze_finetune_updates: int = 0,
    ):
        assert check_argument_types()
        super().__init__()

        if w2v_url != "":
            try:
                import fairseq
                from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
            except Exception as e:
                print("Error: FairSeq is not properly installed.")
                print(
                    "Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done"
                )
                raise e

        self.w2v_model_path = download_w2v(w2v_url, w2v_dir_path)

        self._output_size = output_size

        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.w2v_model_path],
            arg_overrides={"data": w2v_dir_path},
        )
        model = models[0]

        if not isinstance(model, Wav2Vec2Model):
            try:
                model = model.w2v_encoder.w2v_model
            except Exception as e:
                print(
                    "Error: pretrained models should be within: "
                    "'Wav2Vec2Model, Wav2VecCTC' classes, etc."
                )
                raise e

        self.encoders = model

        self.pretrained_params = copy.deepcopy(model.state_dict())

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        if model.cfg.encoder_embed_dim != output_size:
            # TODO(xkc09): try LSTM
            self.output_layer = torch.nn.Sequential(
                torch.nn.Linear(model.cfg.encoder_embed_dim, output_size),
            )
        else:
            self.output_layer = None

        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.LongTensor([0]))

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        return_all_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward FairSeqWav2Vec2 Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
            return_all_hs: Whether to return all hidden states.
                           Unused. Always return None as hidden_states.

        Returns:
            position embedded tensor and mask
        """
        xs_pad, masks = self._forward(xs_pad, ilens)

        bs = xs_pad.shape[0]
        if masks is not None:  # (B, T)
            olens = (~masks).sum(dim=1)  # (B)
        else:
            olens = torch.IntTensor([xs_pad.shape[1]]).repeat(bs).to(xs_pad.device)

        if self.output_layer is not None:
            xs_pad = self.output_layer(xs_pad)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        if return_all_hs:
            xs_pad = (xs_pad, None)
        return xs_pad, olens, None

    def _forward(
            self,
            xs_pad: torch.Tensor,
            ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # fairseq's version doesn't cause OOM
        masks = lengths_to_padding_mask(ilens)

        ft = self.freeze_finetune_updates <= self.num_updates
        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1
        elif ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.info("Start fine-tuning wav2vec parameters!")

        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad() if not ft else contextlib.nullcontext():
                enc_outputs = self.encoders(
                    xs_pad,
                    masks,
                    mask=self.training,
                    features_only=True,
                )

        xs_pad = enc_outputs["x"]  # (B, T, C)
        return xs_pad, enc_outputs["padding_mask"]

    def reload_pretrained_parameters(self):
        self.encoders.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Wav2Vec model parameters reloaded!")


class FairSeqWav2Vec2EncoderWithAdapter(FairSeqWav2Vec2Encoder):
    def __init__(
            self,
            input_size: int,
            w2v_url: str,
            w2v_dir_path: str = "./",
            output_size: int = 256,
            normalize_before: bool = False,
            freeze_finetune_updates: int = 0,
            adaptor_n_layers: int = 1,
            adaptor_kernel_size: int = 3,
            adaptor_stride: int = 2,
            adaptor_layerdrop: float = 0,
            adaptor_layernorm: bool = False,
    ):
        super().__init__(
            input_size,
            w2v_url,
            w2v_dir_path,
            output_size,
            normalize_before,
            freeze_finetune_updates,
        )
        self.adaptor = Conv1dAdaptor(
            self.encoders.cfg.encoder_embed_dim,
            self.encoders.cfg.encoder_embed_dim,
            n_layers=adaptor_n_layers,
            kernel_size=adaptor_kernel_size,
            stride=adaptor_stride,
            layerdrop=adaptor_layerdrop,
            layernorm=adaptor_layernorm,
        )

    def _forward(
            self,
            xs_pad: torch.Tensor,
            ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, padding_mask = super()._forward(xs_pad, ilens)
        x, padding_mask = self.adaptor(x, padding_mask)
        return x, padding_mask


class Conv1dAdaptor(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            n_layers=3,
            kernel_size=3,
            stride=2,
            layerdrop=0.0,
            layernorm=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            nn.Conv1d(
                in_dim if i == 0 else out_dim,
                out_dim * 2,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                )
            for i in range(n_layers)
        )
        self.stride = stride
        self.layerdrop = layerdrop
        self.layernorm = LayerNorm(in_dim) if layernorm else None

    def forward(self, x, padding_mask: Optional[torch.Tensor]):
        if self.layernorm is not None:
            x = self.layernorm(x)

        if padding_mask is not None:
            x = utils.index_put(x, padding_mask, 0)

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)
        out_lens = None
        if padding_mask is not None:
            out_lens = (~padding_mask).sum(1).float()

        for layer in self.layers:
            layerdrop_prob = np.random.random()
            if not self.training or (layerdrop_prob > self.layerdrop):
                x = nn.functional.glu(layer(x), dim=1)
                if padding_mask is not None:
                    out_lens = ((out_lens - 1) / self.stride + 1).floor()

        # B x C x T -> B x T x C
        x = x.transpose(1, 2)

        out_padding_mask = None
        if padding_mask is not None:
            out_padding_mask = lengths_to_padding_mask(out_lens.long())
            x = utils.index_put(x, out_padding_mask, 0)

        return x, out_padding_mask


def download_w2v(model_url, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    model_name = model_url.split("/")[-1]
    model_path = os.path.join(dir_path, model_name)

    dict_url = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt"
    dict_path = os.path.join(dir_path, dict_url.split("/")[-1])

    with FileLock(model_path + ".lock"):
        if not os.path.exists(model_path):
            torch.hub.download_url_to_file(model_url, model_path)
            torch.hub.download_url_to_file(dict_url, dict_path)
            logging.info(f"Wav2Vec model downloaded {model_path}")
        else:
            logging.info(f"Wav2Vec model {model_path} already exists.")

    return model_path
