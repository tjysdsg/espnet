"""
Some of the utility functions copied from or based on
https://github.com/kaldi-asr/kaldi/blob/master/egs/gop_speechocean762/s5/local/utils.py
"""
import argparse
import regex
import json
from speechocean762 import load_human_scores, load_phone_symbol_table


def get_args():
    parser = argparse.ArgumentParser(
        description='Calculate ASE metrics such as FAR, FRR, FER, and more')
    parser.add_argument('hyp', metavar='HYP', type=str,
                        help='Hypothesis file (hyp.trn)')
    parser.add_argument('output', metavar='OUT', type=str, help='Output file path')
    parser.add_argument('--scores', type=str, default='data/local/scores.json', help='Path to scores.json')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    pat = r'([A-Za-z\s]+)\((\S+)-(\S+)\)'
    matcher = regex.compile(pat)

    hyps = {}
    with open(args.hyp) as f:
        for line in f:
            res = matcher.match(line.strip('\n'))
            hyp: str = res.group(1)
            spk: str = res.group(2)
            utt: str = res.group(3)

            # strip leading and trailing SIL
            hyp = hyp.strip(' ').strip('sil')
            hyps[utt] = hyp.split()

    print(hyps)

    scores, phones = load_human_scores(args.scores)
    json.dump(scores, open('tmp/score_of.json', 'w'), indent='\t')
    json.dump(phones, open('tmp/phone_of.json', 'w'), indent='\t')


if __name__ == '__main__':
    main()
