import argparse
import os
from utils import create_logger, load_utt2phones
from speechocean762 import load_human_scores, load_phone_symbol_table, load_so762_ref
from ase_score import get_scores, eval_scoring
import pickle
from sklearn.tree import DecisionTreeRegressor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', metavar='ACTION', choices=['train', 'test'], help='train or test')
    parser.add_argument('hyp', metavar='HYP', type=str, help='Hypothesis file')
    parser.add_argument('ref', metavar='REF', type=str, help='Reference file')
    parser.add_argument('--scores', type=str, help='Path to scores.json')
    parser.add_argument('--phone-table', type=str, help='Path to phones-pure.txt')
    parser.add_argument('--output-dir', type=str, default='tmp', help='Where to save the results')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger = create_logger('train_scoring_model', f'{args.output_dir}/train_scoring_model.log')

    hyps = load_utt2phones(args.hyp)
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

        n = len(utt_align)
        for i in range(n):
            err, i_l, i_p = utt_align[i]
            if err == 'S' or err == '=':
                ppl = ph2int[pred[i_p]]
                cpl = ph2int[label[i_l]]
                x.append([ppl, cpl])
                y.append(sc[i_l])
            elif err == 'D':
                ppl = ph2int['SIL']
                cpl = ph2int[label[i_l]]
                x.append([ppl, cpl])
                y.append(sc[i_l])
            elif err == 'I':
                pass
            else:
                assert False

    logger.info(f'{len(x)} phones in data')
    if args.action == 'train':
        mdl = DecisionTreeRegressor(random_state=42)
        mdl.fit(x, y)
        pickle.dump(mdl, open(os.path.join(args.output_dir, 'scoring.mdl'), 'wb'))
    elif args.action == 'test':
        mdl: DecisionTreeRegressor = pickle.load(open(os.path.join(args.output_dir, 'scoring.mdl'), 'rb'))
        y_pred = mdl.predict(x, y)
        pcc, mse = eval_scoring(y_pred, y)
        logger.info(f'Pearson Correlation Coefficient: {pcc:.4f}')
        logger.info(f'MSE: {mse:.4f}')
    else:
        assert False


if __name__ == '__main__':
    main()
