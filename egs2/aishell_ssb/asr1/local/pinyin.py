from g2pM import G2pM
from typing import List


class _Pinyin:
    INITIALS = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'w', 'x',
                'y', 'z', 'zh']
    EMPTY_PINYIN = ' '


def text2pinyin(text: str) -> List[str]:
    if len(text) == 0:
        return []

    model = G2pM()
    pinyin = model(text, tone=True, char_split=False)

    ret = []
    for p in pinyin:
        ret += to_kaldi_style_pinyin(p)

    return ret


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
