import argparse
import numpy as np
from speechocean762 import load_phone_symbol_table
from scoring_model import load_utt2probs, N_PHONES
from utils import EMPTY_PHONES, load_utt2phones, onehot


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str)
    parser.add_argument('--probs', type=str)
    parser.add_argument('--phone-table', type=str)
    parser.add_argument('--output-path', type=str)
    return parser.parse_args()


def main():
    args = get_args()

    ph2int, int2ph = load_phone_symbol_table(args.phone_table)

    utt2probs = load_utt2probs(args.probs)
    # remove empty phones
    for utt in utt2probs.keys():
        probs = utt2probs[utt]
        new_probs = [p for p in probs if int2ph[np.argmax(p)] not in EMPTY_PHONES]
        utt2probs[utt] = np.asarray(new_probs)

    utt2tokens = load_utt2phones(args.token)
    utt2onehot = {}
    for utt in utt2tokens.keys():
        utt2onehot[utt] = np.asarray([onehot(N_PHONES, ph2int[p]) for p in utt2tokens[utt]])

    # combine
    of = open(args.output_path, 'w')
    for utt in utt2probs.keys():
        assert utt in utt2onehot
        onehots = utt2onehot[utt]
        probs = utt2probs[utt]
        if onehots.shape != probs.shape:
            print(f'onehots and probs of {utt} have different lengths')
            continue

        combined = 0.5 * onehots + 0.5 * probs

        s = str(combined.tolist())
        of.write(f'{utt}\t{s}\n')
    of.close()


if __name__ == '__main__':
    main()
