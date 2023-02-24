import os
from argparse import ArgumentParser
from config import utt2time, utt2spk, utt2story, pwa_spks


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--reco-list", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument("--max-duration", type=float, default=30)
    return parser.parse_args()


def main():
    args = get_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    out_wav_scp = open(os.path.join(args.out_dir, "wav.scp"), "w", encoding="utf-8")
    out_segments = open(os.path.join(args.out_dir, "segments"), "w", encoding="utf-8")
    too_short_path = os.path.join(args.out_dir, "too_short.list")
    too_short = open(too_short_path, "w", encoding="utf-8")
    too_long_path = os.path.join(args.out_dir, "too_long.list")
    too_long = open(too_long_path, "w", encoding="utf-8")

    lengths = []
    n_too_short = 0
    n_too_long = 0
    with open(args.reco_list) as f:
        for line in f:
            seg = line.rstrip("\n")
            spk = utt2spk(seg)
            start, end = utt2time(seg)
            start = float(start) / 1000
            end = float(end) / 1000  # ms to seconds

            if end - start < args.min_duration:
                # print(f"WARNING: {seg} too short (<{args.min_duration}s)")
                n_too_short += 1
                too_short.write(f"{seg}\t{end - start}\n")
                continue

            if end - start > args.max_duration:
                # print(f"WARNING: {seg} too long (>{args.max_duration}s)")
                n_too_long += 1
                too_long.write(f"{seg}\t{end - start}\n")
                continue

            lengths.append(end - start)

            group = "Dementia" if spk in pwa_spks else "Control"
            story = utt2story(seg)
            wav_path = os.path.join(args.data_root, group, story, f"{spk[:-1]}-{spk[-1]}.wav")

            if not os.path.exists(wav_path):
                print(f"WARNING: {wav_path} does not exist")
                continue

            out_wav_scp.write(f"{spk}_{story}\t{wav_path}\n")
            out_segments.write(f"{seg}\t{spk}_{story}\t{start}\t{end}\n")

    print(
        f"WARNING: {n_too_short} utterances are too short"
        f"(<{args.min_duration}s), see {too_short_path}"
    )
    print(
        f"WARNING: {n_too_short} utterances are too long"
        f"(>{args.max_duration}s), see {too_long_path}"
    )


if __name__ == "__main__":
    main()
