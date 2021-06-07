"""
Some of the utility functions copied from or based on
https://github.com/kaldi-asr/kaldi/blob/master/egs/gop_speechocean762/s5/local/utils.py
"""
import argparse
import regex
from utils import remove_sil_from_phone_list
from get_utt2phone import get_utt2phone
from speechocean762 import load_human_scores
from metrics import predict_scores, wer_details_for_batch
from typing import Dict, List
import numpy as np
from scipy.stats import pearsonr

trn_pat = r'([A-Za-z\s]+)\((\S+)-(\S+)\)'
trn_matcher = regex.compile(trn_pat)


def get_args():
    parser = argparse.ArgumentParser(
        description='Calculate ASE correlation between predicted scores and annotated scores')
    parser.add_argument('hyp', metavar='HYP', type=str, help='Hypothesis file')
    parser.add_argument('ref', metavar='REF', type=str, help='Reference file')
    parser.add_argument('output', metavar='OUT', type=str, help='Output file path')
    parser.add_argument('--scores', type=str, default='data/local/scores.json', help='Path to scores.json')
    args = parser.parse_args()
    return args


def load_hypothesis(path: str) -> Dict[str, List[str]]:
    hyps = {}
    with open(path) as f:
        for line in f:
            tokens = line.strip('\n').split()
            utt = tokens[0]
            hyp = tokens[1:]
            phones = remove_sil_from_phone_list(hyp)
            hyps[utt] = phones

    return hyps


def get_scores(hyps: Dict[str, List[str]], refs: Dict[str, List[str]]) -> Dict[str, List[int]]:
    ref_list = []
    hyp_list = []
    utts = []
    for utt, h in hyps.items():
        utts.append(utt)
        hyp_list.append(h)
        ref_list.append(refs[utt])

    details = wer_details_for_batch(utts, ref_list, hyp_list, compute_alignments=True)
    return predict_scores(utts, details)


def main():
    args = get_args()

    hyps = load_hypothesis(args.hyp)
    refs = get_utt2phone(args.ref)
    preds = get_scores(hyps, refs)
    labels, _ = load_human_scores(args.scores)

    hyp_scores = []
    true_scores = []
    for utt, s in preds.items():
        hyp_scores += s
        if utt in labels:
            true_scores += labels[utt]
        else:
            print(f'WARNING: Cannot find annotated score for {utt}')

    pcc, p_test = pearsonr(np.asarray(hyp_scores), np.asarray(true_scores))
    print(f'Pearson Correlation Coefficient: {pcc:.4f}')


if __name__ == '__main__':
    main()
