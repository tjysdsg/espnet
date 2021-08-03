import argparse
from speechocean762 import load_phone_symbol_table
from scoring_model import load_utt2probs
import numpy as np
from typing import List
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probs', type=str, help='Hypothesis file')
    parser.add_argument('--phone-table', type=str, help='Path to phones-pure.txt')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    return parser.parse_args()


def plot_probmat(prob: np.ndarray, phones: List[str], output_path: str):
    from matplotlib import pyplot as plt

    prob = np.clip(np.log(prob), -10, 0)

    fig, ax = plt.subplots()
    fig.set_figheight(20)
    fig.set_figwidth(15)
    ax.set_yticks(np.arange(len(phones)))
    ax.set_yticklabels(phones)

    # im = ax.imshow(prob.T)
    # fig.colorbar(im)
    ax.imshow(prob.T)

    plt.savefig(os.path.join(output_path))
    plt.close('all')


def main():
    args = get_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    ph2int, int2ph = load_phone_symbol_table(args.phone_table)
    utt2probs = load_utt2probs(args.probs)

    phones = [int2ph[i] for i in sorted(list(int2ph.keys()))]
    for utt, probs in utt2probs.items():
        output_path = os.path.join(output_dir, f'{utt}.png')
        plot_probmat(probs, phones, output_path)


if __name__ == '__main__':
    main()
