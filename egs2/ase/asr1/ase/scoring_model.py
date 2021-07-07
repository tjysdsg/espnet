import argparse
import os
from utils import onehot, load_utt2seq
from speechocean762 import load_phone_symbol_table
from ase_score import get_scores, eval_scoring
from typing import Dict, List, Tuple
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import json
from dataclasses import dataclass

# TODO: fix triphone

N_PHONES = 44
SIMILAR_PHONES = [
    ['AA', 'AH'],
    ['AE', 'EH'],
    ['IY', 'IH'],
    ['UH', 'UW'],
]
ph2similar = {}
for group in SIMILAR_PHONES:
    for curr in group:
        ph2similar[curr] = [p for p in group if p != curr]


@dataclass(frozen=True)
class Phone:
    name: str
    probs: np.ndarray = None

    def __hash__(self):
        tup = (self.name, self.probs.tostring() if self.probs is not None else 'None')
        return tup.__hash__()

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __str__(self):
        return f'Phone({self.name})'

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def get_sil_phone() -> 'Phone':
        sil_vec = np.full(N_PHONES, -100)
        sil_vec[0] = 0  # <blank>
        return Phone(name='<blank>', probs=sil_vec)


@dataclass(frozen=True)
class Sample:
    cpl: Phone  # correct phone label
    ppl: Phone  # perceived phone label
    score: int

    def __str__(self):
        return f'{{cpl={self.cpl}, ppl={self.ppl}, score={self.score}}}'

    def __repr__(self):
        return self.__str__()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', metavar='ACTION', choices=['train', 'test'], help='train or test')
    parser.add_argument('hyp', metavar='HYP', type=str, help='Hypothesis file')
    parser.add_argument('ref', metavar='REF', type=str, help='Reference file')
    parser.add_argument('--scores', type=str, help='Path to utt2scores')
    parser.add_argument('--phone-table', type=str, help='Path to phones-pure.txt')
    parser.add_argument('--model-path', type=str, default='tmp/scoring.mdl', help='Where to save the model')
    parser.add_argument('--output-dir', type=str, default='tmp', help='Output directory')
    parser.add_argument('-n', type=int, default=0, help='Number of neighboring phones to include on each side')
    parser.add_argument('--use-probs', action='store_true', default=False,
                        help='Whether HYP contains tokens or probability matrices')
    parser.add_argument('--plot-probs', action='store_true', default=False, help='Plot prob matrices')
    parser.add_argument('--use-mlp', action='store_true', default=False, help='Use neural network model')
    parser.add_argument('--per-phone-clf', action='store_true', default=False, help='Use a model per phone')
    args = parser.parse_args()
    return args


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


def load_utt2phones(path: str) -> Dict[str, List[Phone]]:
    from utils import EMPTY_PHONES

    ret = load_utt2seq(path)
    for utt in ret.keys():
        phones = [Phone(name=ph) for ph in ret[utt] if ph not in EMPTY_PHONES]
        ret[utt] = phones

    return ret


def get_alignment(hyps: Dict[str, List[Phone]], refs: Dict[str, List[Phone]]):
    def to_phone_names(d: Dict[str, List[Phone]]) -> Dict[str, List[str]]:
        ret = {}
        for k, v in d.items():
            names = [e.name for e in v]
            ret[k] = names

        return ret

    _, _, alignment = get_scores(to_phone_names(hyps), to_phone_names(refs))
    return alignment


def load_hyp_ref_score_alignment(
        hyp_path: str, ref_path: str, scores_path: str, use_probs: bool, int2ph: Dict[int, str] = None
) -> Tuple[
    Dict[str, List[Phone]],
    Dict[str, List[Phone]],
    Dict[str, List[int]],
    Dict[str, List],
]:
    # hyps
    if use_probs:
        from utils import EMPTY_PHONES

        hyps = load_utt2probs(hyp_path)
        assert int2ph
        for utt in hyps.keys():
            phones = []
            probs = hyps[utt]
            for p in probs:
                ph_name = int2ph[np.argmax(p)]

                # remove empty phones
                if ph_name not in EMPTY_PHONES:
                    phones.append(Phone(name=ph_name, probs=p))
            hyps[utt] = phones
    else:
        hyps = load_utt2phones(hyp_path)

    # refs, scores
    refs = load_utt2phones(ref_path)
    scores = load_utt2seq(scores_path, formatter=int)

    # get WER alignment
    alignment = get_alignment(hyps, refs)

    return hyps, refs, scores, alignment


def get_utt_samples(preds: List[Phone], labels: List[Phone], scores: List[int], align) -> List[Sample]:
    ret: List[Sample] = []
    n = len(align)
    i_l = 0
    i_p = 0
    for i in range(n):
        err, i1, i2 = align[i]
        if err == 'S' or err == '=':
            assert i_l == i1
            assert i_p == i2

            ppl = preds[i_p]
            cpl = labels[i_l]
            s = scores[i_l]
            i_p += 1
            i_l += 1
        elif err == 'D':
            assert i_l == i1

            ppl = Phone.get_sil_phone()
            cpl = labels[i_l]
            s = scores[i_l]
            i_l += 1
        elif err == 'I':
            assert i_p == i2

            i_p += 1
            continue
        else:
            assert False

        ret.append(Sample(cpl=cpl, ppl=ppl, score=s))
    return ret


def load_data(
        hyp_path: str, ref_path: str, scores_path: str, use_probs: bool, phone_size: int,
        int2ph: Dict[int, str] = None, plot_probs=False
) -> List[Sample]:
    hyps, refs, scores, alignment = load_hyp_ref_score_alignment(hyp_path, ref_path, scores_path, use_probs, int2ph)

    ret = []
    for utt in refs.keys():
        if utt not in hyps:
            continue
        assert utt in alignment and utt in scores, utt

        preds = hyps[utt]
        labels = refs[utt]
        ret += get_utt_samples(preds, labels, scores[utt], alignment[utt])

        if use_probs and plot_probs:
            output_dir = os.path.join(os.path.dirname(hyp_path), 'prob_plots')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{utt}.png')
            phones = [int2ph[i] for i in sorted(list(int2ph.keys()))]
            plot_probmat(np.asarray([pr.probs for pr in preds]), [lab.name for lab in labels], phones, output_path)

    return ret


def data2array(
        data: List[Sample], ph2int: Dict[str, int], use_probs: bool
) -> (np.ndarray, np.ndarray):
    x = []
    y = []
    for d in data:
        if not use_probs:  # if not using probs, the tokens should be converted to onehot encoded vectors
            ppl = onehot(N_PHONES, ph2int[d.ppl.name])
        else:
            ppl = d.ppl.probs

        cpl = onehot(N_PHONES, ph2int[d.cpl.name])

        x.append(np.concatenate([ppl, cpl]))
        y.append(d.score)

    return np.asarray(x), np.asarray(y)


def count_samples_per_score(data: List[Sample]) -> Dict[int, int]:
    ret = {i: 0 for i in range(3)}
    for d in data:
        ret[d.score] += 1
    return ret


class Scorer:
    def __init__(
            self, ph2int: Dict[str, int], use_mlp=False, per_phone=False, use_probs=False,
            model_args: Dict = None
    ):
        self.ph2int = ph2int
        self.use_mlp = use_mlp
        self.per_phone = per_phone
        self.use_probs = use_probs
        self.scalers = {}
        self.clfs = {}
        self.model_args = model_args

    def preprocess_probs(self, key: str, probs: np.ndarray, train_mode: bool):
        from sklearn.preprocessing import StandardScaler

        probs = np.exp(probs)
        if train_mode:
            self.scalers[key] = StandardScaler()
            ret = self.scalers[key].fit_transform(probs)
        else:
            ret = self.scalers[key].transform(probs)

        print(f'Mean of prob matrix of {key}: {np.mean(probs, axis=0)}')
        print(f'Std of prob matrix of {key}: {np.std(probs, axis=0)}')
        return ret

    def _fit_clf(self, key: str, x: np.ndarray, y: np.ndarray):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neural_network import MLPClassifier

        if self.use_mlp:
            self.clfs[key] = MLPClassifier(**self.model_args)
        else:
            self.clfs[key] = DecisionTreeClassifier(**self.model_args)

        self.clfs[key].fit(x, y)

    def _calc_and_print_metrics(self, key: str, preds: np.ndarray, y: np.ndarray):
        pcc, mse = eval_scoring(preds, y)
        acc = accuracy_score(y, preds)
        confusion = confusion_matrix(y, preds)

        print(f'Pearson Correlation Coefficient of {key}: {pcc:.4f}')
        print(f'MSE of {key}: {mse:.4f}')
        print(f'Accuracy of {key}: {acc:.4f}')
        print(f'Confusion matrix of {key}:\n{confusion}')
        return preds, acc, mse, pcc, confusion

    def _test_clf(self, key: str, x: np.ndarray, y: np.ndarray):
        preds = self.clfs[key].predict(x)
        return self._calc_and_print_metrics(key, preds, y)

    def _fit_per_phone(self, ph2samples: Dict[str, List[Sample]]):
        for phone, samples in ph2samples.items():
            x, y = data2array(samples, self.ph2int, self.use_probs)
            if self.use_probs:
                x = self.preprocess_probs(phone, x, True)
            self._fit_clf(phone, x, y)

    def _fit_all(self, data: List[Sample]):
        x, y = data2array(data, self.ph2int, self.use_probs)
        if self.use_probs:
            x = self.preprocess_probs('all', x, True)
        self._fit_clf('all', x, y)

    def fit(self, data):
        if self.per_phone:
            self._fit_per_phone(data)
        else:
            self._fit_all(data)

    def _test_per_phone(self, ph2samples: Dict[str, List[Sample]]):
        y_pred_all = []
        y_all = []
        for phone, samples in ph2samples.items():
            x, y = data2array(samples, self.ph2int, self.use_probs)
            if self.use_probs:
                x = self.preprocess_probs(phone, x, False)
            preds = self._test_clf(phone, x, y)[0]

            y_pred_all.append(preds)
            y_all.append(y)

        y_pred_all = np.concatenate(y_pred_all)
        y_all = np.concatenate(y_all)
        self._calc_and_print_metrics('all', y_pred_all, y_all)

    def test_one(self, data):
        x, y = data2array(data, self.ph2int, self.use_probs)
        if self.use_probs:
            x = self.preprocess_probs('all', x, False)
        self._test_clf('all', x, y)

    def test(self, data):
        if self.per_phone:
            self._test_per_phone(data)
        else:
            self.test_one(data)

    def plot(self, plot_dir: str):
        if not self.use_mlp and self.per_phone:
            for phone, clf in self.clfs.items():
                output_path = os.path.join(plot_dir, f'{phone}.svg')
                plot_decision_tree(clf, output_path)


def main():
    args = get_args()
    model_dir = os.path.dirname(args.model_path)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    ph2int, int2ph = load_phone_symbol_table(args.phone_table)
    data = load_data(
        args.hyp, args.ref, args.scores, args.use_probs, args.n, int2ph=int2ph,
        plot_probs=args.plot_probs
    )

    # remove duplicates from data
    if args.action == 'train' and args.use_probs:
        # removing duplicates for onehot input will leave only several hundreds samples
        data = list(set(data))

    # save samples to file
    of = open(os.path.join(args.output_dir, 'samples'), 'w')
    for sam in data:
        of.write(f'CPL: {sam.cpl}\tPPL: {sam.ppl}\tScore: {sam.score}\n')
    of.close()

    print(f'Score count: {count_samples_per_score(data)}')

    if args.per_phone_clf:
        print('Using per-phone classifiers')
        model_input = {}
        for d in data:
            model_input.setdefault(d.cpl.name, []).append(d)
    else:
        model_input = data

    if args.action == 'train':
        model_args = dict(random_state=42)
        if args.use_mlp:
            model_args.update(
                early_stopping=True, solver='sgd', learning_rate_init=0.001, hidden_layer_sizes=[512], alpha=0.3
            )
        mdl = Scorer(
            ph2int, use_mlp=args.use_mlp, per_phone=args.per_phone_clf, use_probs=args.use_probs,
            model_args=model_args
        )
        mdl.fit(model_input)
        pickle.dump(mdl, open(args.model_path, 'wb'))
    elif args.action == 'test':
        mdl: Scorer = pickle.load(open(args.model_path, 'rb'))
        mdl.test(model_input)

        # # plot decision trees into svg files
        # model_dir = os.path.dirname(args.model_path)
        # tree_plot_dir = os.path.join(model_dir, 'tree_plots')
        # os.makedirs(tree_plot_dir, exist_ok=True)
        # mdl.plot(tree_plot_dir)
    else:
        assert False


if __name__ == '__main__':
    main()
