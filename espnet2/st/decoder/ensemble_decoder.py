# Copyright 2021 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""
import logging
from typing import Any
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet2.asr.decoder.abs_decoder import AbsDecoder


class EnsembleSTDecoder(AbsDecoder, BatchScorerInterface):
    """Base class of Transfomer decoder module.

    Args:
        decoders: ensemble decoders
    """

    def __init__(
        self,
        decoders: List[AbsDecoder],
        is_md_decoders: List[bool] = None,
        md_has_speechattn: List[bool] = None,
        weights: List[float] = None,
    ):
        assert check_argument_types()
        super().__init__()
        assert len(decoders) > 0, "At least one decoder is needed for ensembling"

        # Note (jiatong): different from other'decoders
        self.decoders = decoders
        self.n_decoders = len(self.decoders)
        self.is_md_decoders = is_md_decoders
        self.md_has_speechattn = md_has_speechattn
        self.weights = (
            [1.0 / len(decoders)] * len(decoders) if weights is None else weights
        )
        self.weights = np.array(self.weights)

    def init_state(self, x: torch.Tensor) -> Any:
        """Get an initial state for decoding (optional).

        Args:
            x (torch.Tensor): The encoded feature tensor
        Returns: initial state
        """
        return [None] * len(self.decoders)

    def batch_init_state(self, x: torch.Tensor) -> Any:
        """Get an initial state for decoding (optional).

        Args:
            x (torch.Tensor): The encoded feature tensor
        Returns: initial state
        """
        return self.init_state(x)

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dummy forward"""
        pass

    def score(self, ys, state, x, speech= None):
        """Score."""
        assert len(x) == len(
            self.decoders
        ), "Num of encoder output does not match number of decoders"
        logps = []
        states = []
        for i in range(self.n_decoders):
            ys_mask = subsequent_mask(len(ys), device=x[i].device).unsqueeze(0)
            sub_state = None if state is None else state[i]
            if self.md_has_speechattn[i]:
                logp, sub_state = self.decoders[i].forward_one_step(
                    ys.unsqueeze(0), ys_mask, x[i].unsqueeze(0), cache=sub_state
                )
            else:
                logp, sub_state = self.decoders[i].forward_one_step(
                    ys.unsqueeze(0), ys_mask, x[i].unsqueeze(0), speech[i].unsqueeze(0), cache=sub_state
                )
            #logps.append(np.log(self.weights[i]) + logp.squeeze(0))
            logps.append(self.weights[i] * logp.squeeze(0))
            states.append(sub_state)
        #return torch.logsumexp(torch.stack(logps, dim=0), dim=0), states
        return torch.sum(torch.stack(logps, dim=0), dim=0), states

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: Union[torch.Tensor, List[torch.Tensor]],
        speech=None,
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (Union[torch.Tensor, List[torch.Tensor]]):
                The encoder feature that generates ys (n_batch, xlen, n_feat).
        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.
        """
        n_batch = len(states)

        all_state_list = []
        logps = []
        for i in range(self.n_decoders):
            decoder_batch = [states[h][i] for h in range(n_batch)]
            if self.md_has_speechattn[i]:
                logp, state_list = self.decoders[i].batch_score(ys, decoder_batch, xs[i], speech[i])
            else:
                logp, state_list = self.decoders[i].batch_score(ys, decoder_batch, speech[i])
            all_state_list.append(state_list)
            logps.append(self.weights[i] * logp)
        score = torch.sum(torch.stack(logps, dim=0), dim=0)

        transpose_state_list = []
        for i in range(n_batch):
            transpose_state_list.append(
                [all_state_list[j][i] for j in range(self.n_decoders)]
            )
        return score, transpose_state_list
