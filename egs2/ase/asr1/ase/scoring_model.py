import argparse
import os
import random

from aug import add_more_negative_data
from utils import load_utt2phones, onehot, load_utt2seq
from speechocean762 import load_phone_symbol_table
from ase_score import get_scores, eval_scoring
from typing import Dict, List, Any, Tuple, Callable
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import json

N_PHONES = 44
SIL_VEC = np.full(N_PHONES, -100)  # <blank>
SIL_VEC[0] = 0


class NPhone:
    def __init__(self, curr: str, left: List[str], right: List[str]):
        self.left = left
        self.right = right
        self.curr = curr

    def __str__(self):
        return "{" + f'{self.curr}, {self.left}, {self.right}' + "}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def tolist(self):
        return self.left + [self.curr] + self.right


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', metavar='ACTION', choices=['train', 'test'], help='train or test')
    parser.add_argument('hyp', metavar='HYP', type=str, help='Hypothesis file')
    parser.add_argument('ref', metavar='REF', type=str, help='Reference file')
    parser.add_argument('--balance', type=bool, default=True, help='Balance data, only used for training')
    parser.add_argument('--scores', type=str, help='Path to utt2scores')
    parser.add_argument('--phone-table', type=str, help='Path to phones-pure.txt')
    parser.add_argument('--model-path', type=str, default='tmp/scoring.mdl', help='Where to save the model')
    parser.add_argument('--output-dir', type=str, default='tmp', help='Output directory')
    parser.add_argument('-n', type=int, default=0, help='Number of neighboring phones to include on each side')
    parser.add_argument('--use-probs', action='store_true', default=False,
                        help='Whether HYP contains tokens or probability matrices')
    parser.add_argument('--plot-probs', action='store_true', default=False, help='Plot prob matrices')
    parser.add_argument('--downsample-extra-data', action='store_true', default=False, help='Plot prob matrices')
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

    refs = load_utt2phones(ref_path)
    scores = load_utt2seq(scores_path, formatter=int)

    _, _, alignment = get_scores(hyps, refs)

    ph2data = {}
    for utt in refs.keys():
        if utt not in hyps:
            continue

        assert utt in alignment and utt in scores, utt
        label = refs[utt]
        utt_align = alignment[utt]
        pred = hyps[utt]
        sc = scores[utt]

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
    def __init__(self, phone_names: List[str], use_mlp: bool, per_phone: bool, use_probs: bool, *args, **kwargs):
        self.use_mlp = use_mlp
        self.per_phone = per_phone
        self.use_probs = use_probs
        if per_phone:
            self.scalers = {p: StandardScaler() for p in phone_names}
            if use_mlp:
                self.clfs = {p: MLPClassifier(*args, **kwargs) for p in phone_names}
            else:
                self.clfs = {p: DecisionTreeClassifier(*args, **kwargs) for p in phone_names}
        else:
            self.scalers = StandardScaler()
            if use_mlp:
                self.clfs = MLPClassifier(*args, **kwargs)
            else:
                self.clfs = DecisionTreeClassifier(*args, **kwargs)

    def __getitem__(self, phone: str):
        return self.clfs[phone]

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def preprocess_probs(self, probs, train_mode: bool, phone: str = None):
        if self.per_phone:
            scaler = self.scalers[phone]
        else:
            scaler = self.scalers

        if train_mode:
            ret = scaler.fit_transform(np.exp(probs))
        else:
            ret = scaler.transform(np.exp(probs))

        print(f'Mean of prob matrix of phone {phone}: {np.mean(probs, axis=0)}')
        print(f'Std of prob matrix of phone {phone}: {np.std(probs, axis=0)}')
        return ret

    def fit(self, data):
        if self.per_phone:
            for phone, samples in data.items():
                x, y = samples
                if self.use_probs:
                    x = self.preprocess_probs(x, True, phone)
                self.clfs[phone].fit(x, y)
        else:
            x, y = data
            if self.use_probs:
                x = self.preprocess_probs(x, True)
            self.clfs.fit(x, y)

    def test_per_phone(self, ph2samples: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        y_pred_all = []
        y_all = []
        for phone, samples in ph2samples.items():
            x, y = samples
            if self.use_probs:
                x = self.preprocess_probs(x, False, phone)
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
        if self.use_probs:
            x = self.preprocess_probs(x, False)
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

    # remove duplicates from data
    for ph in ph2data.keys():
        phonedata = [tuple(d) for d in ph2data[ph]]
        ph2data[ph] = list(set(phonedata))

    # save ph2data to file
    of = open(os.path.join(args.output_dir, 'ph2data'), 'w')
    for ph, d in ph2data.items():
        for ppl, cpl, s in d:
            of.write(f'CPL {cpl}\tPPL: {ppl}\tScore: {s}\n')
    of.close()

    score_count = {i: 0 for i in range(3)}
    score2data = {}
    for ph in ph2data.keys():
        x, y = to_data_samples(ph2data[ph], ph2int, args.use_probs)
        for i, s in enumerate(y):
            score2data.setdefault(s, []).append((ph, x[i], y[i]))
            score_count[s] += 1

    # save score2data to file, with phone index converted to phone names
    of = open(os.path.join(args.output_dir, 'score2data'), 'w')
    for s, d in score2data.items():
        for p, x, _ in d:
            of.write(f'CPL: {p}\tPPL: {int2ph[np.argmax(x)]}\tScore: {s}\n')
    of.close()

    # downsample some of the data so that the number of samples with different scores are the same
    N = np.min(list(score_count.values()))
    ph2data = {}
    for s in score2data.keys():
        d = score2data[s]

        if args.downsample_extra_data:
            nd = len(d)
            idx = list(range(nd))
            idx = random.sample(idx, N)
            d = [d[i] for i in idx]

        for ph, x, y in d:
            ph2data.setdefault(ph, []).append((x, y))

    # flatten ph2data
    if args.per_phone_clf:
        print('Using per-phone classifiers')
        data = {}
    else:
        data = [[], []]
    score_count = {i: 0 for i in range(3)}
    for ph, d in ph2data.items():
        xs = np.asarray([_d[0] for _d in d])
        ys = np.asarray([_d[1] for _d in d])
        if args.per_phone_clf:
            data[ph] = (xs, ys)
        else:
            data[0].append(xs)
            data[1].append(ys)
        for s in ys:
            score_count[s] += 1
    print(f'Score counts: {score_count}')
    if not args.per_phone_clf:
        data[0] = np.vstack(data[0])
        data[1] = np.concatenate(data[1])

    model_dir = os.path.dirname(args.model_path)
    os.makedirs(model_dir, exist_ok=True)

    phone_names = list(ph2data.keys())
    if args.action == 'train':
        other_args = {}
        if args.use_mlp:
            other_args = dict(
                early_stopping=True, solver='sgd', learning_rate_init=0.001, hidden_layer_sizes=[512], alpha=0.3,
                # verbose=True,
            )

        mdl = Scorer(
            phone_names, random_state=42, use_mlp=args.use_mlp, per_phone=args.per_phone_clf,
            use_probs=args.use_probs, **other_args
        )
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
