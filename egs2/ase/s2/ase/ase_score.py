"""
Some of the utility functions copied from or based on
https://github.com/kaldi-asr/kaldi/blob/master/egs/gop_speechocean762/s5/local/utils.py
"""
import argparse
import os
from utils import load_utt2phones, load_utt2seq
from metrics import get_wer_details
from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
from dataclasses import dataclass
import collections


@dataclass
class ScorePhone:
    phone: str
    score: int


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str)
    parser.add_argument('--cpl', type=str)
    parser.add_argument('--utt2scores', type=str)
    parser.add_argument('--output-dir', type=str, help='Where to save the results')
    args = parser.parse_args()
    return args


def get_wer_alignment(hyps: Dict[str, List[str]], refs: Dict[str, List[str]]) -> Dict:
    details = get_wer_details(hyps, refs)

    # {utt -> wer alignments}
    wer_align = {}
    for d in details:
        wer_align[d['key']] = d['alignment']

    return wer_align


def get_phone_pairs(wer_align: List, hyp: List[str], ref: List[str], label: List[float]) -> (List[str], List[str]):
    label = [str(int(score)) for score in label]

    n = len(wer_align)
    preds = []
    labels = []
    indices = [0 for _ in range(2)]
    for i in range(n):
        err = wer_align[i][0]
        if err == 'S' or err == '=':
            preds.append(hyp[indices[0]])
            labels.append(ref[indices[1]] + label[indices[1]])
            indices[0] += 1
            indices[1] += 1
        elif err == 'I':
            preds.append(hyp[indices[0]])
            labels.append(' ')
            indices[0] += 1
        elif err == 'D':
            preds.append(' ')
            labels.append(ref[indices[1]] + label[indices[1]])
            indices[1] += 1
        else:
            assert False

    return preds, labels


def get_pred_label(wer_align: List, pred: List[ScorePhone], label: List[ScorePhone]) -> (List[int], List[int]):
    n = len(wer_align)
    ret_pred = []
    ret_label = []
    pred_idx = 0
    label_idx = 0
    for i in range(n):
        err = wer_align[i][0]

        if err == 'S' or err == '=':
            pred_sp = pred[pred_idx]
            true_sp = label[label_idx]

            if pred_sp.phone != true_sp.phone:
                ret_pred.append(0)
            else:
                ret_pred.append(pred_sp.score)

            ret_label.append(true_sp.score)
            pred_idx += 1
            label_idx += 1
        elif err == 'I':  # pred has an insertion, ignore it
            pred_idx += 1
        elif err == 'D':  # label has an insertion, pred score is 0
            ret_pred.append(0)
            ret_label.append(label[label_idx].score)
            label_idx += 1
        else:
            assert False

    return ret_pred, ret_label


def eval_scoring(pred: np.ndarray, true: np.ndarray) -> (float, float):
    pcc, _ = pearsonr(pred, true)
    mse = mean_squared_error(pred, true)
    print(f'Pearson Correlation Coefficient: {pcc:.4f}')
    print(f'MSE: {mse:.4f}')

    print(f'Acc: {accuracy_score(true, pred)}')
    print(f'Confusion:\n{confusion_matrix(true, pred)}')


def find_most_incorrect_sp_pairs(phone_pairs: List[Tuple[str, str]]):
    pairs = []  # all incorrect sp pairs
    for pred, true in phone_pairs:
        pred_phone = pred[:-1]
        true_phone = true[:-1]

        if '0' in true and pred_phone == true_phone:  # e.g. 'IH1' 'IH0'
            pairs.append((pred, true))

        if '0' not in true and pred != true:  # e.g. 'IY2' 'IY1'
            pairs.append((pred, true))

    counts = collections.Counter(pairs)
    print(counts.most_common(20))


def get_score_phones(phones: List[str]) -> List[ScorePhone]:
    return [ScorePhone(phone=sp[:-1], score=int(sp[-1])) for sp in phones]


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    hyps = load_utt2phones(args.hyp)
    cpls = load_utt2phones(args.cpl)
    utt2scores = load_utt2seq(args.utt2scores, formatter=int)

    hyps_phones = {k: [p[:-1] for p in v] for k, v in hyps.items()}  # phones without scores attached
    wer_align = get_wer_alignment(hyps_phones, cpls)

    hyp_scores = []
    true_scores = []
    sp_pairs = []
    f = open(f'{args.output_dir}/alignment.txt', 'w')
    for utt, s in hyps.items():
        pred = get_score_phones(s)

        if utt in cpls:
            alignment = wer_align[utt]

            utt_scores = utt2scores[utt]
            assert len(utt_scores) == len(cpls[utt])
            cpl = [ScorePhone(phone=p, score=utt_scores[i]) for i, p in enumerate(cpls[utt])]

            pred, label = get_pred_label(alignment, pred, cpl)

            hyp_phones, ref_phones = get_phone_pairs(alignment, hyps[utt], cpls[utt], label)
            n = len(hyp_phones)
            assert n == len(ref_phones)

            # write to alignment.txt
            f.write(f'utt: {utt}\n')
            f.write('\t'.join(hyp_phones) + '\n')
            f.write('\t'.join(ref_phones) + '\n\n')

            sp_pairs += list(zip(hyp_phones, ref_phones))

            hyp_scores += pred
            true_scores += label
        else:
            print(f'WARNING: Cannot find annotated score for {utt}')

    f.close()

    eval_scoring(np.asarray(hyp_scores), np.asarray(true_scores))
    find_most_incorrect_sp_pairs(sp_pairs)


if __name__ == '__main__':
    main()
