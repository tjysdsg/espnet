import random
from collections import Counter
from typing import Dict, List

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


def add_more_negative_data(data: Dict[str, List]):
    """
    Take the 2-score samples of other phones as new 0-score samples, and take the 2-score samples of
    other similar phones as new 1-score samples
    """
    ph2different_samples = {}  # {ph: [samples that contains phone != ph]
    ph2similar_samples = {}  # {ph: [samples that contains phone that is similar to ph]
    for curr_ph in data.keys():
        ph2different_samples.setdefault(curr_ph, [])
        ph2similar_samples.setdefault(curr_ph, [])
        for ph, feats in data.items():
            if ph != curr_ph:
                sam = [[f[0], f[1], 0] for f in feats if f[-1] == 2]
                ph2different_samples[curr_ph] += sam

                if ph in ph2similar.get(curr_ph, []):
                    sam = [[f[0], f[1], 1] for f in feats if f[-1] == 2]
                    ph2similar_samples[curr_ph] += sam

    for curr_ph in data:
        ppls, cpls, scores = list(zip(*data[curr_ph]))
        score_counts = Counter(scores)
        n_samples_needed = 2 * score_counts[2] - len(cpls)

        # generate 0-score samples
        if n_samples_needed > 0:
            data[curr_ph] += random.sample(ph2different_samples[curr_ph], n_samples_needed)

        # generate 1-score samples
        if 0 < n_samples_needed:
            similar_samples = ph2similar_samples[curr_ph]
            data[curr_ph] += random.sample(
                similar_samples,
                min(n_samples_needed, len(similar_samples))
            )

    return data
