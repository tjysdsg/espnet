"""
Some of the utility functions copied from or based on
https://github.com/kaldi-asr/kaldi/blob/master/egs/gop_speechocean762/s5/local/utils.py
"""
import argparse
import os
from utils import load_utt2phones, create_logger
from metrics import wer_details_for_batch
from typing import Dict, List
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass


@dataclass
class ScorePhone:
    phone: str
    score: int


def get_args():
    parser = argparse.ArgumentParser(
        description='Calculate ASE correlation between predicted scores and annotated scores')
    parser.add_argument('--hyp', type=str, help='Hypothesis file')
    parser.add_argument('--ref', type=str, help='Reference file')
    parser.add_argument('--output-dir', type=str, help='Where to save the results')
    args = parser.parse_args()
    return args


def get_wer_alignment(hyps: Dict[str, List[str]], refs: Dict[str, List[str]]) -> Dict:
    ref_list = []
    hyp_list = []
    utts = []
    for utt, h in hyps.items():
        utts.append(utt)
        hyp_list.append(h)
        ref_list.append(refs[utt])

    details = wer_details_for_batch(utts, ref_list, hyp_list, compute_alignments=True)

    # {utt -> wer alignments}
    wer_align = {}
    for d in details:
        wer_align[d['key']] = d['alignment']

    return wer_align


def get_result_str(wer_align: List, hyp: List[str], ref: List[str], label: List[float]) -> str:
    label = [str(int(score)) for score in label]

    n = len(wer_align)
    lines = ['' for _ in range(3)]
    indices = [0 for _ in range(3)]
    for i in range(n):
        err = wer_align[i][0]
        if err == 'S' or err == '=':
            lines[0] += '\t' + hyp[indices[0]]
            lines[1] += '\t' + ref[indices[1]]
            lines[2] += '\t' + label[indices[2]]
            indices[0] += 1
            indices[1] += 1
            indices[2] += 1
        elif err == 'I':
            lines[0] += '\t' + hyp[indices[0]]
            lines[1] += '\t '
            lines[2] += '\t '
            indices[0] += 1
        elif err == 'D':
            lines[0] += '\t '
            lines[1] += '\t' + ref[indices[1]]
            lines[2] += '\t' + label[indices[2]]
            indices[1] += 1
            indices[2] += 1
        else:
            assert False

    return f'pred_phones:\t{lines[0]}\n' \
           f'true_phones:\t{lines[1]}\n' \
           f'true_scores:\t{lines[2]}\n'


def get_pred_label(wer_align: List, pred: List[ScorePhone], label: List[ScorePhone]) -> (List[int], List[int]):
    n = len(wer_align)
    ret_pred = []
    ret_label = []
    pred_idx = 0
    label_idx = 0
    for i in range(n):
        err = wer_align[i][0]

        if err == 'S' or err == '=':
            pred_sp = pred[pred_idx]
            true_sp = label[label_idx]

            if pred_sp.phone != true_sp.phone:
                ret_pred.append(0)
            else:
                ret_pred.append(pred_sp.score)

            ret_label.append(true_sp.score)
            pred_idx += 1
            label_idx += 1
        elif err == 'I':  # pred has an insertion, ignore it
            pred_idx += 1
        elif err == 'D':  # label has an insertion, pred score is 0
            ret_pred.append(0)
            ret_label.append(label[label_idx].score)
            label_idx += 1
        else:
            assert False

    return ret_pred, ret_label


def eval_scoring(pred: np.ndarray, true: np.ndarray) -> (float, float):
    pcc, _ = pearsonr(pred, true)
    mse = mean_squared_error(pred, true)
    return pcc, mse


def get_score_phones(phones: List[str]) -> List[ScorePhone]:
    return [ScorePhone(phone=sp[:-1], score=int(sp[-1])) for sp in phones]


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger = create_logger('ase_score', f'{args.output_dir}/ase_score.log')

    hyps = load_utt2phones(args.hyp)
    refs = load_utt2phones(args.ref)
    wer_align = get_wer_alignment(hyps, refs)

    hyp_scores = []
    true_scores = []
    f = open(f'{args.output_dir}/alignment.txt', 'w')
    for utt, s in hyps.items():
        pred = get_score_phones(s)
        if utt in refs:
            label = get_score_phones(refs[utt])
            alignment = wer_align[utt]

            pred, label = get_pred_label(alignment, pred, label)

            # FIXME:
            #   f.write(f'utt: {utt}\n')
            #   f.write(get_result_str(alignment, hyps[utt], refs[utt], label))

            hyp_scores += pred
            true_scores += label
        else:
            print(f'WARNING: Cannot find annotated score for {utt}')

    f.close()

    pcc, mse = eval_scoring(np.asarray(hyp_scores), np.asarray(true_scores))
    logger.info(f'Pearson Correlation Coefficient: {pcc:.4f}')
    logger.info(f'MSE: {mse:.4f}')


if __name__ == '__main__':
    main()
