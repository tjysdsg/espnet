from g2pM import G2pM
from typing import List


class _Pinyin:
    INITIALS = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'w', 'x',
                'y', 'z', 'zh']
    EMPTY_PINYIN = ' '
    PHONES = {"a", "ai", "an", "ang", "ao", "b", "c", "ch", "d", "e", "ei", "en", "eng", "er", "f", "g", "h", "i", "ia",
              "ian", "iang", "iao", "ie", "in", "ing", "io", "iong", "iu", "j", "k", "l", "m", "n", "o", "ong", "ou",
              "p", "q", "r", "s", "sh", "t", "u", "ua", "uai", "uan", "uang", "uao", "ue", "ui", "un", "uo", "v", "ve",
              "w", "x", "y", "z", "zh"}


def text2pinyin(text: str) -> List[str]:
    if len(text) == 0:
        return []

    model = G2pM()
    pinyin = model(text, tone=True, char_split=False)

    ret = []
    for p in pinyin:
        ret += to_kaldi_style_pinyin(p)

    ret = [p for p in ret if is_phone_valid(p)]
    return ret


def is_phone_valid(p: str):
    tokens = p.split('_')
    if len(tokens) == 1:  # initials
        return tokens[0] in _Pinyin.PHONES
    elif len(tokens) == 2:  # finals, must have a tone
        return tokens[0] in _Pinyin.PHONES and tokens[1].isdigit()
    else:
        return False


def to_kaldi_style_pinyin(pinyin: str) -> List[str]:
    if pinyin == _Pinyin.EMPTY_PINYIN:
        return []
    if pinyin[:-1] == 'r':  # 儿化
        pinyin = 'er' + '_' + pinyin[-1]
    else:
        pinyin = pinyin[:-1] + '_' + pinyin[-1]
    ret = []
    len_ = len(pinyin)
    assert len_ > 0
    final_start = 0
    if len_ >= 2 and pinyin[:2] in _Pinyin.INITIALS:
        ret.append(pinyin[:2])
        final_start = 2
    elif pinyin[0] in _Pinyin.INITIALS:
        ret.append(pinyin[0])
        final_start = 1

    ret.append(pinyin[final_start:])
    return ret


if __name__ == '__main__':
    print(text2pinyin('LAUGH_'))
