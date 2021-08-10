import argparse
import librosa


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wavscp', type=str)
    return parser.parse_args()


def main():
    args = get_args()

    total_sec = 0.

    with open(args.wavscp) as f:
        for line in f:
            utt, wav = line.strip('\n').split()

            total_sec += librosa.get_duration(filename=wav)

    print(f'{total_sec / 3600} hours')


if __name__ == '__main__':
    main()
