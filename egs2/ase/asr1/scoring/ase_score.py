"""
Some of the utility functions copied from or based on
https://github.com/kaldi-asr/kaldi/blob/master/egs/gop_speechocean762/s5/local/utils.py
"""
import argparse
import regex
import json
from speechocean762 import load_human_scores, load_phone_symbol_table
from metrics import predict_scores, wer_details_for_batch
from typing import Dict, List

trn_pat = r'([A-Za-z\s]+)\((\S+)-(\S+)\)'
trn_matcher = regex.compile(trn_pat)


def get_args():
    parser = argparse.ArgumentParser(
        description='Calculate ASE correlation between predicted scores and annotated scores')
    parser.add_argument('hyp', metavar='HYP', type=str, help='Hypothesis file (hyp.trn)')
    parser.add_argument('ref', metavar='REF', type=str, help='Reference file (ref.trn)')
    parser.add_argument('output', metavar='OUT', type=str, help='Output file path')
    parser.add_argument('--scores', type=str, default='data/local/scores.json', help='Path to scores.json')
    args = parser.parse_args()
    return args


def load_trn_file(path: str) -> Dict[str, List[str]]:
    hyps = {}
    with open(path) as f:
        for line in f:
            res = trn_matcher.match(line.strip('\n'))
            hyp: str = res.group(1)
            # spk: str = res.group(2)
            utt: str = res.group(3)

            # strip leading and trailing SIL
            hyp = hyp.strip(' ').strip('sil')
            hyps[utt] = hyp.split()

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

    hyps = load_trn_file(args.hyp)
    refs = load_trn_file(args.ref)
    scores = get_scores(hyps, refs)
    print(scores)

    # scores, phones = load_human_scores(args.scores)
    # json.dump(scores, open('tmp/score_of.json', 'w'), indent='\t')
    # json.dump(phones, open('tmp/phone_of.json', 'w'), indent='\t')


if __name__ == '__main__':
    main()
