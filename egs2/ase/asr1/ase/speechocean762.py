import os
import json
from utils import convert_to_pure_phones


def load_phone_symbol_table(filename):
    if not os.path.isfile(filename):
        return None, None
    int2sym = {}
    sym2int = {}
    with open(filename, 'r') as f:
        for line in f:
            sym, idx = line.strip('\n').split('\t')
            idx = int(idx)
            int2sym[idx] = sym
            sym2int[sym] = idx
    return sym2int, int2sym


def round_score(score, floor=0.1, min_val=0, max_val=2):
    score = max(min(max_val, score), min_val)
    return round(score / floor) * floor


def load_human_scores(filename, floor=0.1):
    with open(filename) as f:
        info = json.load(f)
    score_of = {}
    phone_of = {}
    for utt in info:
        phone_of[utt] = []
        score_of[utt] = []
        for word in info[utt]['words']:
            assert len(word['phones']) == len(word['phones-accuracy'])
            for i, phone in enumerate(word['phones']):
                pure_phone = convert_to_pure_phones(phone)
                s = round_score(word['phones-accuracy'][i], floor)
                phone_of[utt].append(pure_phone)
                score_of[utt].append(s)
    return score_of, phone_of
