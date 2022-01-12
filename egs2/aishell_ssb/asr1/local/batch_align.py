import argparse
import os
from espnet2.bin.asr_align import CTCSegmentation
import librosa


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-train-config", type=str)
    parser.add_argument("--asr-model-file", type=str)
    parser.add_argument("--wavscp", type=str)
    parser.add_argument("--text", type=str)
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--fs", type=int, default=16000)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    utt2path = {}
    with open(args.wavscp) as f:
        for line in f:
            utt, path = line.strip('\n').split()
            utt2path[utt] = path

    utt2phones = {}
    with open(args.text) as f:
        for line in f:
            tokens = line.strip('\n').split()
            utt = tokens[0]
            phones = tokens[1:]
            utt2phones[utt] = phones

    utts = list(set(utt2path.keys()))

    # perform alignment
    config = dict(
        asr_train_config=args.asr_train_config,
        asr_model_file=args.asr_model_file,
        fs=args.fs,
        kaldi_style_text=False,
    )
    aligner = CTCSegmentation(**config)

    with open(os.path.join(args.out_dir, f'{utt}.txt'), 'w') as f:
        for utt in utts:
            wav, sr = librosa.load(utt2path[utt], sr=args.fs)
            text = '\n'.join(utt2phones[utt])

            try:
                segments = aligner(wav, text)
            except AssertionError as e:
                print(f'{utt} failed')
                continue

            f.write(f'{str(segments)}\n')


if __name__ == "__main__":
    main()
