import argparse
import os
import random
from collections import Counter
from utils import load_utt2phones, onehot
from speechocean762 import load_human_scores, load_phone_symbol_table, load_so762_ref
from ase_score import get_scores, eval_scoring
from typing import Dict, List, Any, Tuple
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import json

# TODO: refactor

N_PHONES = 44
SIL_VEC = np.full(N_PHONES, -100)  # <blank>
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
    parser.add_argument('--balance', action='store_true', default=True, help='Balance data, only used for training')
    parser.add_argument('--scores', type=str, help='Path to scores.json')
    parser.add_argument('--phone-table', type=str, help='Path to phones-pure.txt')
    parser.add_argument('--model-path', type=str, default='tmp/scoring.mdl', help='Where to save the results')

    parser.add_argument('-n', type=int, default=0, help='Number of neighboring phones to include on each side')
    parser.add_argument('--use-probs', action='store_true', default=False,
                        help='Whether HYP contains tokens or probability matrices')

    parser.add_argument('--plot-probs', action='store_true', default=False, help='Plot prob matrices')
    parser.add_argument('--use-mlp', action='store_true', default=False, help='Use neural network model')
    parser.add_argument('--per-phone-clf', action='store_true', default=False, help='Use a model per phone')
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
                ph2other_ph.setdefault(curr_ph, [])
                ph2other_ph[curr_ph] += [[f[0], f[1], 0] for f in feats if f[-1] == 2]

    # take the 2-score examples of other phones as the negative examples
    for curr_ph in data:
        ppls, cpls, scores = list(zip(*data[curr_ph]))
        count_of_label = Counter(scores)
        example_number_needed = 2 * count_of_label[2] - len(cpls)

        if example_number_needed > 0:
            data[curr_ph] += random.sample(ph2other_ph[curr_ph], example_number_needed)

    return data


def to_data_samples(
        phone_data: List[NPhone or int], ph2int: Dict[str, int], use_probs: bool
) -> (np.ndarray, np.ndarray):
    x = []
    y = []
    for d in phone_data:
        ppl, cpl, s = d
        if not use_probs:  # if not using probs, the tokens should be converted to onehot encoded vectors
            ppl = [onehot(N_PHONES, ph2int[p]) for p in ppl.tolist()]
        else:
            ppl = ppl.tolist()
        cpl = [onehot(N_PHONES, ph2int[p]) for p in cpl.tolist()]

        x.append(np.asarray(ppl + cpl).ravel())
        y.append(s)

    return np.asarray(x), np.asarray(y)


def plot_probmat(prob: np.ndarray, labels: List[str], phones: List[str], output_path: str):
    from matplotlib import pyplot as plt

    prob = np.clip(prob, -10, 10)  # clip large values so the colors are shown properly

    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(len(phones)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(phones)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation='vertical')
    ax.margins(0.2)

    ax.imshow(prob)
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
            phones = [int2ph[i] for i in sorted(list(int2ph.keys()))]
            plot_probmat(np.asarray(pred), label, phones, output_path)

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


def plot_decision_tree(mdl, output_path: str):
    """
    Plot decision tree structure into an svg file
    """
    from sklearn import tree
    from matplotlib import pyplot as plt
    plt.figure(figsize=(19, 10), dpi=130)
    tree.plot_tree(mdl)
    plt.savefig(output_path)
    plt.close('all')


class Scorer:
    def __init__(self, phone_names: List[str], use_mlp: bool, per_phone: bool, *args, **kwargs):
        self.use_mlp = use_mlp
        self.per_phone = per_phone
        if per_phone:
            if use_mlp:
                self.clfs = {p: MLPClassifier(*args, **kwargs) for p in phone_names}
            else:
                self.clfs = {p: DecisionTreeClassifier(*args, **kwargs) for p in phone_names}
        else:
            if use_mlp:
                self.clfs = MLPClassifier(*args, **kwargs)
            else:
                self.clfs = DecisionTreeClassifier(*args, **kwargs)

    def __getitem__(self, phone: str):
        return self.clfs[phone]

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def fit(self, data):
        if self.per_phone:
            for phone, samples in data.items():
                x, y = samples
                self.clfs[phone].fit(x, y)
        else:
            x, y = data
            self.clfs.fit(x, y)

    def test_per_phone(self, ph2samples: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        y_pred_all = []
        y_all = []
        for phone, samples in ph2samples.items():
            x, y = samples
            y_pred = self.clfs[phone].predict(x)

            print(f'Accuracy of phone {phone}: {accuracy_score(y, y_pred):.4f}')
            print(f'Confusion matrix of phone {phone}:\n{confusion_matrix(y, y_pred)}')

            y_pred_all.append(y_pred)
            y_all.append(y)

        y_pred_all = np.concatenate(y_pred_all)
        y_all = np.concatenate(y_all)
        pcc, mse = eval_scoring(y_pred_all, y_all)
        print('=' * 40)
        print(f'Overall Pearson Correlation Coefficient: {pcc:.4f}')
        print(f'Overall MSE: {mse:.4f}')
        print(f'Overall Accuracy: {accuracy_score(y_all, y_pred_all):.4f}')
        print(f'Overall confusion matrix:\n{confusion_matrix(y_all, y_pred_all)}')
        print('=' * 40)

    def test_one(self, data):
        x, y = data
        y_pred = self.clfs.predict(x)

        pcc, mse = eval_scoring(y_pred, y)
        print('=' * 40)
        print(f'Pearson Correlation Coefficient: {pcc:.4f}')
        print(f'MSE: {mse:.4f}')
        print(f'Accuracy: {accuracy_score(y, y_pred):.4f}')
        print(f'Confusion matrix:\n{confusion_matrix(y, y_pred)}')
        print('=' * 40)

    def test(self, data):
        if self.per_phone:
            self.test_per_phone(data)
        else:
            self.test_one(data)

    def plot(self, plot_dir: str):
        if not self.use_mlp and self.per_phone:
            for phone, clf in self.clfs.items():
                output_path = os.path.join(plot_dir, f'{phone}.svg')
                plot_decision_tree(clf, output_path)


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
        print('Performing data augmentation')
        ph2data = add_more_negative_data(ph2data)

    if args.per_phone_clf:
        print('Using per-phone classsifiers')
        data = {}
    else:
        data = [[], []]

    score_count = {}
    for ph, d in ph2data.items():
        x, y = to_data_samples(d, ph2int, args.use_probs)

        if args.per_phone_clf:
            data[ph] = (x, y)
        else:
            data[0].append(x)
            data[1].append(y)

        for s in y:
            score_count.setdefault(s, 0)
            score_count[s] += 1

    if not args.per_phone_clf:
        data[0] = np.vstack(data[0])
        data[1] = np.concatenate(data[1])

    print(f'Score counts: {score_count}')

    model_dir = os.path.dirname(args.model_path)
    os.makedirs(model_dir, exist_ok=True)

    phone_names = list(ph2data.keys())
    if args.action == 'train':
        mlp_args = dict(
            early_stopping=True, solver='sgd', learning_rate_init=0.001, hidden_layer_sizes=[512], alpha=0.3,
            # verbose=True,
        )
        if args.use_mlp:
            mdl = Scorer(phone_names, random_state=42, use_mlp=True, per_phone=args.per_phone_clf, **mlp_args)
        else:
            mdl = Scorer(phone_names, random_state=42, use_mlp=False, per_phone=args.per_phone_clf)

        mdl.fit(data)
        pickle.dump(mdl, open(args.model_path, 'wb'))
    elif args.action == 'test':
        mdl: Scorer = pickle.load(open(args.model_path, 'rb'))
        mdl.test(data)

        # plot decision trees into svg files
        model_dir = os.path.dirname(args.model_path)
        tree_plot_dir = os.path.join(model_dir, 'tree_plots')
        os.makedirs(tree_plot_dir, exist_ok=True)
        mdl.plot(tree_plot_dir)
    else:
        assert False


if __name__ == '__main__':
    main()
