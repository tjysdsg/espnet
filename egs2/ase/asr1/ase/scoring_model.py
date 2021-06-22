import argparse
from utils import load_utt2phones, onehot
from speechocean762 import load_human_scores, load_phone_symbol_table, load_so762_ref
from ase_score import get_scores, eval_scoring
from typing import Dict, List
import pickle
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import json

N_PHONES = 44
SIL_VEC = np.zeros(N_PHONES)  # FIXME


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', metavar='ACTION', choices=['train', 'test'], help='train or test')
    parser.add_argument('hyp', metavar='HYP', type=str, help='Hypothesis file')
    parser.add_argument('ref', metavar='REF', type=str, help='Reference file')
    parser.add_argument('-n', type=int, default=0, help='Number of neighboring phones to include on each side')
    parser.add_argument('--use-probs', action='store_true', default=False,
                        help='Whether HYP contains tokens or probability matrices')
    parser.add_argument('--scores', type=str, help='Path to scores.json')
    parser.add_argument('--phone-table', type=str, help='Path to phones-pure.txt')
    parser.add_argument('--model-path', type=str, default='tmp/scoring.mdl', help='Where to save the results')
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
            probs = json.loads(s)[1:]  # FIXME: the first one is always <sos>
            probs = np.asarray(probs)
            hyps[utt] = probs

    return hyps


def main():
    args = get_args()

    if args.use_probs:
        hyps = load_utt2probs(args.hyp)
        SIL = SIL_VEC
    else:
        hyps = load_utt2phones(args.hyp)
        SIL = 'SIL'

    refs = load_so762_ref(args.ref)
    ph2int, _ = load_phone_symbol_table(args.phone_table)
    scores, _ = load_human_scores(args.scores)

    _, _, alignment = get_scores(hyps, refs)

    x = []
    y = []
    for utt in hyps.keys():
        assert utt in refs and utt in alignment and utt in scores

        label = refs[utt]
        utt_align = alignment[utt]
        pred = hyps[utt]
        sc = scores[utt]

        def try_get_phone(phones, idx):
            if idx < 0 or idx >= len(phones):
                return SIL if args.use_probs else 'SIL'
            else:
                return phones[idx]

        def get_phone_grams(phones, idx: int, is_deletion=False):
            # if is_deletion is true, the current position between idx-1 and idx
            size = args.n
            left = [try_get_phone(phones, i) for i in range(idx - size, idx)]
            if is_deletion:
                right = [try_get_phone(phones, i) for i in range(idx, idx + size)]
            else:
                right = [try_get_phone(phones, i) for i in range(idx + 1, idx + size + 1)]

            if is_deletion:
                ret = left + [SIL] + right
            else:
                ret = left + [try_get_phone(phones, idx)] + right
            return ret

        n = len(utt_align)
        i_l = 0
        i_p = 0
        for i in range(n):
            err, i1, i2 = utt_align[i]
            if err == 'S' or err == '=':
                assert i_l == i1
                assert i_p == i2

                ppl = get_phone_grams(pred, i_p)
                cpl = label[i_l]
                s = sc[i_l]
                i_p += 1
                i_l += 1
            elif err == 'D':
                assert i_l == i1

                ppl = get_phone_grams(pred, i_p, is_deletion=True)
                cpl = label[i_l]
                s = sc[i_l]
                i_l += 1
            elif err == 'I':
                assert i_p == i2

                i_p += 1
                continue
            else:
                assert False

            if not args.use_probs:
                ppl = [onehot(N_PHONES, ph2int[p]) for p in ppl]
            cpl = onehot(N_PHONES, ph2int[cpl])
            x.append(np.asarray(ppl + [cpl]).ravel())
            y.append(s)

    if args.action == 'train':
        mdl = DecisionTreeRegressor(random_state=42)
        mdl.fit(x, y)
        pickle.dump(mdl, open(args.model_path, 'wb'))
    elif args.action == 'test':
        mdl: DecisionTreeRegressor = pickle.load(open(args.model_path, 'rb'))
        y_pred = mdl.predict(x, y)
        pcc, mse = eval_scoring(y_pred, y)
        print(f'Pearson Correlation Coefficient: {pcc:.4f}')
        print(f'MSE: {mse:.4f}')
    else:
        assert False


if __name__ == '__main__':
    main()
