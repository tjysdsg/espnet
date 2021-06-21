"""
Some of the utility functions copied from or based on
https://github.com/kaldi-asr/kaldi/blob/master/egs/gop_speechocean762/s5/local/utils.py
"""
import argparse
import os
from utils import load_utt2phones, create_logger
from speechocean762 import load_human_scores, load_so762_ref
from metrics import predict_scores, wer_details_for_batch, wer_summary
from typing import Dict, List
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def get_args():
    parser = argparse.ArgumentParser(
        description='Calculate ASE correlation between predicted scores and annotated scores')
    parser.add_argument('hyp', metavar='HYP', type=str, help='Hypothesis file')
    parser.add_argument('ref', metavar='REF', type=str, help='Reference file')
    parser.add_argument('--scores', type=str, default=None, help='Path to scores.json')
    parser.add_argument('--output-dir', type=str, default='tmp', help='Where to save the results')
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


def get_scores(hyps: Dict[str, List[str]], refs: Dict[str, List[str]]) -> (Dict, Dict[str, List[int]], Dict):
    ref_list = []
    hyp_list = []
    utts = []
    for utt, h in hyps.items():
        utts.append(utt)
        hyp_list.append(h)
        ref_list.append(refs[utt])

    details = wer_details_for_batch(utts, ref_list, hyp_list, compute_alignments=True)
    wer = wer_summary(details)

    # {utt -> wer alignments}
    wer_align = {}
    for d in details:
        wer_align[d['key']] = d['alignment']

    return wer, predict_scores(utts, details), wer_align


def get_result_str(wer_align: List, hyp: List[str], ref: List[str], pred: List[float], label: List[float]) -> str:
    pred = [str(int(score)) for score in pred]
    label = [str(int(score)) for score in label]

    n = len(wer_align)
    lines = ['' for _ in range(4)]
    indices = [0 for _ in range(3)]
    for i in range(n):
        err = wer_align[i][0]
        if err == 'S' or err == '=':
            lines[0] += '\t' + hyp[indices[0]]
            lines[1] += '\t' + ref[indices[1]]
            lines[2] += '\t' + pred[indices[2]]
            lines[3] += '\t' + label[indices[2]]
            indices[0] += 1
            indices[1] += 1
            indices[2] += 1
        elif err == 'I':
            lines[0] += '\t' + hyp[indices[0]]
            lines[1] += '\t '
            lines[2] += '\t '
            lines[3] += '\t '
            indices[0] += 1
        elif err == 'D':
            lines[0] += '\t '
            lines[1] += '\t' + ref[indices[1]]
            lines[2] += '\t' + pred[indices[2]]
            lines[3] += '\t' + label[indices[2]]
            indices[1] += 1
            indices[2] += 1
        else:
            assert False

    return f'pred_phones:\t{lines[0]}\n' \
           f'true_phones:\t{lines[1]}\n' \
           f'pred_scores:\t{lines[2]}\n' \
           f'true_scores:\t{lines[3]}\n'


def eval_scoring(score_pred: List[float], score_true: List[float]) -> (float, float):
    x1 = np.asarray(score_pred)
    x2 = np.asarray(score_true)
    pcc, _ = pearsonr(x1, x2)
    mse = mean_squared_error(x1, x2)
    return pcc, mse


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger = create_logger('ase_score', f'{args.output_dir}/ase_score.log')

    hyps = load_utt2phones(args.hyp)
    refs = load_so762_ref(args.ref)
    wer, preds, wer_align = get_scores(hyps, refs)

    logger.info(wer)

    labels, _ = load_human_scores(args.scores)
    f = open(f'{args.output_dir}/alignment.txt', 'w')
    hyp_scores = []
    true_scores = []
    for utt, s in preds.items():
        hyp_scores += s
        if utt in labels:
            true_scores += labels[utt]
            error_type = wer_align[utt]
            f.write(f'utt: {utt}\n')
            f.write(get_result_str(error_type, hyps[utt], refs[utt], preds[utt], labels[utt]))
        else:
            print(f'WARNING: Cannot find annotated score for {utt}')

    f.close()

    pcc, mse = eval_scoring(hyp_scores, true_scores)
    logger.info(f'Pearson Correlation Coefficient: {pcc:.4f}')
    logger.info(f'MSE: {mse:.4f}')


if __name__ == '__main__':
    main()
