import os
import json
from utils import convert_to_pure_phones
from typing import Dict, List


def load_phone_symbol_table(filename: str) -> (Dict[str, int], Dict[int, str]):
    if not os.path.isfile(filename):
        return None, None
    int2sym = {}
    sym2int = {}
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            sym = line.strip('\n')
            int2sym[i] = sym
            sym2int[sym] = i
    return sym2int, int2sym


def load_so762_ref(text_phone: str) -> Dict[str, List[str]]:
    """
    Return a dictionary containing the correct phone transcripts of speechocean762 utterances

    :param text_phone The text-phone file in so762 dataset
    :return {utt: [phone1, phone2, ...]}
    """
    u2t2p = {}
    with open(text_phone) as f:
        for line in f:
            tokens = line.replace('\n', '').split()
            assert len(tokens) > 1
            utt = tokens[0]
            trans = tokens[1:]

            assert '.' in utt
            utt, time = utt.split('.')
            time = int(time)
            if utt not in u2t2p:
                u2t2p[utt] = {time: trans}
            else:
                u2t2p[utt][time] = trans

    # sort phones by time
    u2p = {}
    for utt, t2p in u2t2p.items():
        temporal = sorted(t2p.items(), key=lambda x: x[0])
        phones = [p for _, ph in temporal for p in ph]
        u2p[utt] = [convert_to_pure_phones(p) for p in phones]

    return u2p


def round_score(score, floor=0.1, min_val=0, max_val=2):
    score = max(min(max_val, score), min_val)
    return round(score / floor) * floor


def load_utt2cpl_ppl_scores(scores_json: str, floor=1):
    """
    Note that PPL is <DEL> if there is a deletion error, <UNK>
    """
    with open(scores_json) as f:
        info = json.load(f)

    utt2cpl = {}
    utt2ppl = {}
    utt2scores = {}
    for utt in info:
        utt2cpl[utt] = []
        utt2ppl[utt] = []
        utt2scores[utt] = []
        for word in info[utt]['words']:
            assert len(word['phones']) == len(word['phones-accuracy'])

            ppl = []
            for i, phone in enumerate(word['phones']):
                # cpl
                pure_phone = convert_to_pure_phones(phone)
                utt2cpl[utt].append(pure_phone)

                # use cpl as ppl at first, then overwrite it with mispronunciation
                ppl.append(pure_phone)

                # score
                s = round_score(word['phones-accuracy'][i], floor)
                utt2scores[utt].append(s)

            # ppl
            for mispron in word.get('mispronunciations', []):
                idx = mispron['index']
                ppl[idx] = mispron['pronounced-phone'].upper()
            utt2ppl[utt] += ppl

    return utt2cpl, utt2ppl, utt2scores
