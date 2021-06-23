import argparse
import os
import random
from collections import Counter
from utils import load_utt2phones, onehot
from speechocean762 import load_human_scores, load_phone_symbol_table, load_so762_ref
from ase_score import get_scores, eval_scoring
from typing import Dict, List, Any
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import json

N_PHONES = 44
SIL_VEC = np.full(N_PHONES, -100)  # FIXME
SIL_VEC[0] = 0


class NPhone:
    def __init__(self, curr: str, left: List[str], right: List[str]):
        self.left = left
        self.right = right
        self.curr = curr

    def __repr__(self):
        return f'{self.curr}, {self.left}, {self.right}'

    def tolist(self):
        return self.left + [self.curr] + self.right


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', metavar='ACTION', choices=['train', 'test'], help='train or test')
    parser.add_argument('hyp', metavar='HYP', type=str, help='Hypothesis file')
    parser.add_argument('ref', metavar='REF', type=str, help='Reference file')
    parser.add_argument('-n', type=int, default=0, help='Number of neighboring phones to include on each side')
    parser.add_argument('--use-probs', action='store_true', default=False,
                        help='Whether HYP contains tokens or probability matrices')
    parser.add_argument('--balance', action='store_true', default=True, help='Balance data, only used for training')
    parser.add_argument('--scores', type=str, help='Path to scores.json')
    parser.add_argument('--phone-table', type=str, help='Path to phones-pure.txt')
    parser.add_argument('--model-path', type=str, default='tmp/scoring.mdl', help='Where to save the results')
    parser.add_argument('--plot-probs', action='store_true', default=False, help='Plot prob matrices')
    args = parser.parse_args()
    return args


def load_utt2probs(path: str) -> Dict[str, np.ndarray]:
    """
    Load utt2phones into a dictionary

    :return {utt: [phone1, phone2, ...]}
    """
    hyps = {}
    with open(path) as f:
        for line in f:
            tokens = line.strip('\n').split(maxsplit=1)
            utt = tokens[0]
            s = tokens[1]
            probs = json.loads(s)[1:]  # FIXME: the first one is always invalid because of modified batch_beam_search.py
            probs = np.asarray(probs)
            hyps[utt] = probs

    return hyps


def add_more_negative_data(data: Dict[str, List]):
    ph2other_ph = {}  # {ph: [samples that contains phone != ph]
    for curr_ph in data.keys():
        for ph, feats in data.items():
            if ph != curr_ph:
                ph2other_ph[curr_ph] = [[f[0], f[1], 0] for f in feats if f[-1] == 2]

    # take the 2-score examples of other phones as the negative examples
    for curr_ph in data:
        ppls, cpls, scores = list(zip(*data[curr_ph]))
        labels = [cpl.curr for cpl in cpls]
        count_of_label = Counter(labels)
        example_number_needed = 2 * count_of_label[2] - len(labels)
        if example_number_needed > 0:
            data[curr_ph] += random.sample(ph2other_ph[curr_ph], example_number_needed)

    return data


def to_data_samples(ph2data: Dict[str, List], ph2int: Dict[str, int], use_probs: bool) -> (np.ndarray, np.ndarray):
    x = []
    y = []
    for data in ph2data.values():
        for d in data:
            ppl, cpl, s = d
            if not use_probs:  # if not using probs, the tokens should be converted to onehot encoded vectors
                ppl = [onehot(N_PHONES, ph2int[p]) for p in ppl.tolist()]
            else:
                ppl = ppl.tolist()
            cpl = [onehot(N_PHONES, ph2int[p]) for p in cpl.tolist()]

            x.append(np.asarray(ppl + cpl).ravel())
            y.append(s)

    return np.asarray(x), np.asarray(y)


def plot_probmat(prob: np.ndarray, int2ph: Dict[int, str], output_path: str):
    from matplotlib import pyplot as plt
    labels = np.argmax(prob, axis=1)
    labels = [int2ph[i] for i in labels]

    prob = np.clip(prob, -10, 10)
    plt.imshow(prob)
    plt.savefig(os.path.join(output_path))
    plt.close('all')


def load_data(
        hyp_path: str, ref_path: str, scores_path: str, use_probs: bool, phone_size: int,
        int2ph: Dict[int, str] = None, plot_probs=False
) -> Dict[str, List]:
    if use_probs:
        hyps = load_utt2probs(hyp_path)
        assert int2ph
        # remove empty phones
        for utt in hyps.keys():
            phones = hyps[utt]
            phones = [p for p in phones if int2ph[np.argmax(p)] not in ['<sos/eos>', '<blank>', '<unk>']]
            hyps[utt] = np.asarray(phones)
    else:
        hyps = load_utt2phones(hyp_path)

    refs = load_so762_ref(ref_path)
    scores_path, _ = load_human_scores(scores_path, floor=1)

    _, _, alignment = get_scores(hyps, refs)

    ph2data = {}
    for utt in hyps.keys():
        assert utt in refs and utt in alignment and utt in scores_path

        label = refs[utt]
        utt_align = alignment[utt]
        pred = hyps[utt]
        sc = scores_path[utt]

        if use_probs and plot_probs:
            output_dir = os.path.join(os.path.dirname(hyp_path), 'prob_plots')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{utt}.png')
            plot_probmat(np.asarray(pred), int2ph, output_path)

        def try_get_phone(phones: List[str], idx, sil: str):
            if idx < 0 or idx >= len(phones):
                return sil
            else:
                return phones[idx]

        def get_nphone(phones, idx: int, is_deletion=False, sil: Any = '<blank>') -> NPhone:
            # if is_deletion is true, the current position between idx-1 and idx
            left = [try_get_phone(phones, i, sil) for i in range(idx - phone_size, idx)]
            if is_deletion:
                right = [try_get_phone(phones, i, sil) for i in range(idx, idx + phone_size)]
            else:
                right = [try_get_phone(phones, i, sil) for i in range(idx + 1, idx + phone_size + 1)]

            if is_deletion:
                curr = sil
            else:
                curr = try_get_phone(phones, idx, sil)

            return NPhone(curr, left, right)

        n = len(utt_align)
        i_l = 0
        i_p = 0
        for i in range(n):
            err, i1, i2 = utt_align[i]
            if err == 'S' or err == '=':
                assert i_l == i1
                assert i_p == i2

                ppl = get_nphone(pred, i_p, sil=SIL_VEC if use_probs else '<blank>')
                cpl = get_nphone(label, i_l)
                s = sc[i_l]
                i_p += 1
                i_l += 1
            elif err == 'D':
                assert i_l == i1

                ppl = get_nphone(pred, i_p, is_deletion=True, sil=SIL_VEC if use_probs else '<blank>')
                cpl = get_nphone(label, i_l)
                s = sc[i_l]
                i_l += 1
            elif err == 'I':
                assert i_p == i2

                i_p += 1
                continue
            else:
                assert False

            ph2data.setdefault(cpl.curr, []).append([ppl, cpl, s])  # [perceived nphone, correct nphone, score]

    return ph2data


def main():
    args = get_args()

    ph2int, int2ph = load_phone_symbol_table(args.phone_table)

    # of = open('D:/repos/espnet/tmp/hyp_prob_tokens.txt', 'w')
    # hyp_probs = load_utt2probs(args.hyp)
    # for utt in hyp_probs.keys():
    #     phones = hyp_probs[utt]
    #     phones = [int2ph[np.argmax(h)] for h in phones]
    #     phones = [h for h in phones if h not in ['<sos/eos>', '<blank>', '<unk>']]
    #     phones = " ".join(phones)
    #     of.write(f'{utt}\t{phones}\n')
    # of.close()

    ph2data = load_data(
        args.hyp, args.ref, args.scores, args.use_probs, args.n, int2ph=int2ph,
        plot_probs=args.plot_probs
    )

    if args.action == 'train' and args.balance:
        ph2data = add_more_negative_data(ph2data)

    x, y = to_data_samples(ph2data, ph2int, args.use_probs)
    if args.action == 'train':
        mdl = DecisionTreeClassifier(random_state=42)
        mdl.fit(x, y)
        pickle.dump(mdl, open(args.model_path, 'wb'))
    elif args.action == 'test':
        mdl: DecisionTreeClassifier = pickle.load(open(args.model_path, 'rb'))

        # plot decision tree
        from sklearn import tree
        from matplotlib import pyplot as plt
        plt.figure(figsize=(19, 10), dpi=130)
        tree.plot_tree(mdl)
        plt.savefig('exp/tree.svg')
        plt.close('all')

        y_pred = mdl.predict(x)
        pcc, mse = eval_scoring(y_pred, y)
        print(f'Pearson Correlation Coefficient: {pcc:.4f}')
        print(f'MSE: {mse:.4f}')
        print(f'Accuracy: {accuracy_score(y, y_pred)}')
        print(confusion_matrix(y, y_pred))
    else:
        assert False


if __name__ == '__main__':
    main()
