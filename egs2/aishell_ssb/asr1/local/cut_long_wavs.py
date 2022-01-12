import math
import os
import argparse
import librosa
import json
import soundfile
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-dur', type=int, default=10, help='Max duration in seconds')
    parser.add_argument('--fs', type=int, default=16000)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--out-dir', type=str)
    return parser.parse_args()


def main():
    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    out_wav_dir = os.path.join(out_dir, 'wav')
    os.makedirs(out_wav_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, 'utt2align.json')) as f:
        utt2align: dict = json.load(f)

    # prepare new utt2align.json
    utt2align_clean = {}
    for utt, align in utt2align.items():
        align_clean = []
        curr_dur = 0
        curr_seg = 0
        for start, end, text in align:
            if len(align_clean) > 0 and curr_dur > args.max_dur:
                utt2align_clean[f'{utt}.{curr_seg}'] = align_clean

                align_clean = [
                    (start, end, text)
                ]
                curr_dur = end - start
                curr_seg += 1
            else:
                align_clean.append((start, end, text))
                curr_dur += end - start

        if len(align_clean) > 0:  # save any leftovers
            utt2align_clean[f'{utt}.{curr_seg}'] = align_clean

    with open(os.path.join(out_dir, 'utt2align.json'), 'w') as f:
        json.dump(utt2align_clean, f, indent='  ', ensure_ascii=False)

    # cut long wavs
    utt2path = {}
    with open(os.path.join(args.data_dir, 'wav.scp')) as f:
        for line in f:
            utt, path = line.strip('\n').split()
            utt2path[utt] = path

    text_of = open(os.path.join(out_dir, 'text'), 'w', buffering=1)
    wavscp_of = open(os.path.join(out_dir, 'wav.scp'), 'w', buffering=1)
    cache = {}
    for utt, align in utt2align_clean.items():
        path = utt2path[utt.split('.')[0]]

        if path in cache:
            wav = cache[path]
        else:
            wav, _ = librosa.load(path, sr=args.fs)
            cache[path] = wav

        wav_segs = []
        text_clean = []
        for start, end, text in align:
            s, e = librosa.time_to_samples([start, end], sr=args.fs)
            wav_segs.append(wav[s:e])
            text_clean.append(text)
        wav_clean = np.concatenate(wav_segs)
        text_clean = ' '.join(text_clean)

        # save the new wav file
        out_path = os.path.join(out_wav_dir, f'{utt}.wav')
        soundfile.write(out_path, wav_clean, samplerate=args.fs)

        # update wav.scp
        wavscp_of.write(f'{utt}\t{out_path}\n')

        # update text
        text_of.write(f'{utt}\t{text_clean}\n')

    text_of.close()
    wavscp_of.close()


if __name__ == '__main__':
    main()
