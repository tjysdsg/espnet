import argparse
import numpy as np
from scoring_model import load_utt2probs, load_phone_symbol_table
from utils import load_utt2seq, remove_empty_phones


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, help='utt2phones file')
    parser.add_argument('--probs', type=str, help='utt2probs file')
    parser.add_argument('--phone-table', type=str, help='Path to phones-pure.txt')
    return parser.parse_args()


def main():
    args = get_args()
    utt2probs = load_utt2probs(args.probs)
    utt2phones = load_utt2seq(args.token)
    ph2int, int2ph = load_phone_symbol_table(args.phone_table)

    for utt in utt2probs.keys():
        if utt not in utt2phones:
            print(f'Cannot find {utt} in {args.token}')
            continue

        probs = utt2probs[utt]
        phones = utt2phones[utt]

        argmax = np.argmax(probs, axis=1).ravel().tolist()
        prob_phones = [int2ph[e] for e in argmax]
        prob_phones = remove_empty_phones(prob_phones)

        n = len(prob_phones)
        assert n == len(phones)
        for i in range(n):
            assert prob_phones[i] == phones[i]


if __name__ == '__main__':
    main()
