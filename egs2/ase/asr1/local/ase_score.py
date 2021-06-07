"""
Some of the utility functions copied from or based on
https://github.com/kaldi-asr/kaldi/blob/master/egs/gop_speechocean762/s5/local/utils.py
"""
import os
import argparse
import json
import regex


def get_args():
    parser = argparse.ArgumentParser(
        description='Calculate ASE metrics such as FAR, FRR, FER, and more')
    parser.add_argument('hyp', metavar='HYP', type=str,
                        help='Hypothesis file (hyp.trn)')
    parser.add_argument('output', metavar='OUT', type=str, help='Output file path')
    parser.add_argument('--scores', type=str, default='data/local/scores.json', help='Path to scores.json')
    args = parser.parse_args()
    return args


def load_phone_symbol_table(filename):
    if not os.path.isfile(filename):
        return None, None
    int2sym = {}
    sym2int = {}
    with open(filename, 'r') as f:
        for line in f:
            sym, idx = line.strip('\n').split('\t')
            idx = int(idx)
            int2sym[idx] = sym
            sym2int[sym] = idx
    return sym2int, int2sym


def round_score(score, floor=0.1, min_val=0, max_val=2):
    score = max(min(max_val, score), min_val)
    return round(score / floor) * floor


def load_human_scores(filename, floor=0.1):
    with open(filename) as f:
        info = json.load(f)
    score_of = {}
    phone_of = {}
    for utt in info:
        phone_num = 0
        for word in info[utt]['words']:
            assert len(word['phones']) == len(word['phones-accuracy'])
            phone_of[utt] = []
            score_of[utt] = []
            for i, phone in enumerate(word['phones']):
                pure_phone = regex.sub(r'[_\d].*', '', phone)
                s = round_score(word['phones-accuracy'][i], floor)
                phone_of[utt].append(pure_phone)
                score_of[utt].append(s)
    return score_of, phone_of


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
    # json.dump(scores, open('tmp/score_of.json', 'w'), indent='\t')
    # json.dump(phones, open('tmp/phone_of.json', 'w'), indent='\t')


if __name__ == '__main__':
    main()
