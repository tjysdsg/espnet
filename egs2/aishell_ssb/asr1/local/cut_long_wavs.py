import os
import argparse
import librosa
import json
import soundfile
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-dur', type=float, default=10, help='Max duration in seconds')
    parser.add_argument('--min-dur', type=float, default=3, help='Min duration in seconds, segments that are shorter '
                                                                 'than this will be concatenated to its previous '
                                                                 'segment')
    parser.add_argument('--fs', type=int, default=16000)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--out-dir', type=str)
    parser.add_argument('--only-text', action='store_true', help='Do not output wave files if true')
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
        for start, end, text, words in align:
            if len(align_clean) > 0 and curr_dur > args.max_dur:
                utt2align_clean[f'{utt}.{curr_seg}'] = align_clean

                align_clean = [
                    (start, end, text, words)
                ]
                curr_dur = end - start
                curr_seg += 1
            else:
                align_clean.append((start, end, text, words))
                curr_dur += end - start

        if len(align_clean) > 0:  # save any leftovers
            dur_left = 0
            for a in align_clean:
                dur_left += a[1] - a[0]

            if dur_left < args.min_dur:  # append leftovers to the last segment if it's too short tho
                utt2align_clean[f'{utt}.{curr_seg - 1}'] += align_clean
            else:  # else save the leftovers as a new segment
                utt2align_clean[f'{utt}.{curr_seg}'] = align_clean

    with open(os.path.join(out_dir, 'utt2align.json'), 'w') as f:
        json.dump(utt2align_clean, f, indent='  ', ensure_ascii=False)

    utt2path = {}
    with open(os.path.join(args.data_dir, 'wav.scp')) as f:
        for line in f:
            utt, path = line.strip('\n').split()
            utt2path[utt] = path

    utt2spk = {}
    with open(os.path.join(args.data_dir, 'utt2spk')) as f:
        for line in f:
            utt, spk = line.strip('\n').split()
            utt2spk[utt] = spk

    text_of = open(os.path.join(out_dir, 'text'), 'w', buffering=1)
    wavscp_of = open(os.path.join(out_dir, 'wav.scp'), 'w', buffering=1)
    utt2spk_of = open(os.path.join(out_dir, 'utt2spk'), 'w', buffering=1)
    all_phones = set()

    # cut long wavs
    cache = {}
    for utt, align in utt2align_clean.items():
        orig_utt = utt.split('.')[0]
        path = utt2path[orig_utt]

        if not args.only_text:
            if path in cache:
                wav = cache[path]
            else:
                wav, _ = librosa.load(path, sr=args.fs)
                cache[path] = wav

        wav_segs = []
        text_clean = []
        for start, end, text, _ in align:
            if not args.only_text:
                s, e = librosa.time_to_samples([start, end], sr=args.fs)
                wav_segs.append(wav[s:e])

            text_clean.append(text)

        if not args.only_text:
            wav_clean = np.concatenate(wav_segs)
            print(f'duration of {utt} = {len(wav_clean) / args.fs}')

        text_clean = ' '.join(text_clean)
        for p in text_clean.split():
            all_phones.add(p)

        # save the new wav file
        out_path = os.path.join(out_wav_dir, f'{utt}.wav')
        if not args.only_text:
            soundfile.write(out_path, wav_clean, samplerate=args.fs)

        # update wav.scp
        wavscp_of.write(f'{utt}\t{out_path}\n')

        # update text
        text_of.write(f'{utt}\t{text_clean}\n')

        # utt2spk
        spk = utt2spk[orig_utt]
        utt2spk_of.write(f'{utt}\t{spk}\n')

    text_of.close()
    wavscp_of.close()
    utt2spk_of.close()

    with open(os.path.join(out_dir, 'phones.txt'), 'w') as f:
        for p in all_phones:
            f.write(f'{p}\n')


if __name__ == '__main__':
    main()
