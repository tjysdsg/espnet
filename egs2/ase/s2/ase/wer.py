"""
Some of the utility functions copied from or based on
https://github.com/kaldi-asr/kaldi/blob/master/egs/gop_speechocean762/s5/local/utils.py
"""
import argparse
import os
from utils import remove_empty_phones, create_logger
from metrics import wer_details_for_batch, wer_summary
from typing import Dict, List


def get_args():
    parser = argparse.ArgumentParser(description='Calculate WER')
    parser.add_argument('hyp', metavar='HYP', type=str, help='Hypothesis file')
    parser.add_argument('ref', metavar='REF', type=str, help='Reference file')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    args = parser.parse_args()
    return args


def load_text(path: str) -> Dict[str, List[str]]:
    ret = {}
    with open(path) as f:
        for line in f:
            tokens = line.strip('\n').split()
            utt = tokens[0]
            phones = tokens[1:]
            phones = remove_empty_phones(phones)
            ret[utt] = phones

    return ret


def get_wer(hyps: Dict[str, List[str]], refs: Dict[str, List[str]]) -> (Dict, Dict[str, List[int]], Dict):
    ref_list = []
    hyp_list = []
    utts = []
    for utt, h in hyps.items():
        utts.append(utt)
        hyp_list.append(h)
        ref_list.append(refs[utt])

    details = wer_details_for_batch(utts, ref_list, hyp_list, compute_alignments=True)
    return wer_summary(details)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger = create_logger('wer', f'{args.output_dir}/wer.log')

    hyps = load_text(args.hyp)
    refs = load_text(args.ref)
    wer = get_wer(hyps, refs)

    logger.info(wer)


if __name__ == '__main__':
    main()
