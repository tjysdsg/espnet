import sys
from typing import Dict, List
from utils import convert_to_pure_phones


def get_utt2phone(text_phone: str) -> Dict[str, List[str]]:
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


def main():
    text_phone = sys.argv[1]
    print(get_utt2phone(text_phone))


if __name__ == '__main__':
    main()
