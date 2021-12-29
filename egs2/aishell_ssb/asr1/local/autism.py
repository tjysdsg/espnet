import os
import argparse
import librosa
import json
import soundfile

import numpy as np

ROLE = 'K'  # we only need children data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=str)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--out-dir', type=str)
    return parser.parse_args()


def main():
    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    utts = []
    with open(args.filter) as f:
        for line in f:
            utts.append(line.strip('\n'))

    # utt2align
    align_dir = os.path.join(args.data_dir, 'Mark')
    utt2align = {}
    # all_durs = []
    for utt in utts:
        with open(os.path.join(align_dir, f'{utt}.txt'), encoding='utf-8') as f:
            for line in f:
                role, start, end, text = line.strip('\n').split(maxsplit=3)
                start, end = float(start), float(end)

                if role != ROLE:
                    continue

                utt2align.setdefault(utt, []).append([start, end, text])

                # all_durs.append(end - start)
    json.dump(utt2align, open(os.path.join(out_dir, 'utt2align.json'), 'w'), indent=' ', ensure_ascii=False)

    """
    from matplotlib import pyplot as plt
    dur_min = np.quantile(all_durs, 0.2)
    dur_max = np.quantile(all_durs, 0.8)
    plt.hist(np.clip(all_durs, dur_min, dur_max))
    plt.savefig(os.path.join(out_dir, 'duration.png'))
    plt.close('all')
    """

    # cut waves
    wav_dir = os.path.join(args.data_dir, '16k')
    wav_clean_dir = os.path.join(out_dir, 'wav')
    os.makedirs(wav_clean_dir, exist_ok=True)
    for utt in utts:
        waveform, sr = librosa.load(os.path.join(wav_dir, f'{utt}.wav'))
        cleaned = np.zeros_like(waveform)
        for start, end, _ in utt2align[utt]:
            start, end = librosa.time_to_samples([start, end], sr=sr)
            cleaned[start:end] = waveform[start:end]
        soundfile.write(os.path.join(wav_clean_dir, f'{utt}.wav'), cleaned, samplerate=sr)

    # text
    with open(os.path.join(out_dir, 'text'), 'w') as f:
        for utt in utts:
            text = []
            for _, _, t in utt2align[utt]:
                text.append(t)
            f.write(f'{utt}\t{" ".join(text)}\n')

    # wav.scp
    with open(os.path.join(out_dir, 'wav.scp'), 'w') as f:
        for utt in utts:
            path = os.path.join(wav_clean_dir, f'{utt}.wav')
            f.write(f'{utt}\t{path}\n')

    # utt2spk
    with open(os.path.join(out_dir, 'utt2spk'), 'w') as f:
        for utt in utts:
            f.write(f'{utt}\t{0}\n')
    # spk2utt
    with open(os.path.join(out_dir, 'spk2utt'), 'w') as f:
        utt_str = ' '.join(utts)
        f.write(f'{0}\t{utt_str}\n')


if __name__ == '__main__':
    main()
