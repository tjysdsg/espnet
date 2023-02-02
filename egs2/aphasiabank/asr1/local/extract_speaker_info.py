"""
Based on https://github.com/monirome/AphasiaBank/blob/main/clean_transcriptions.ipynb
"""
import json
import os
from argparse import ArgumentParser

import pylangacq as pla

# these speakers' aphasia type information is missing, but we actually know for sure if they have Aphasia or not
spk2missing_aphasia_types = {
    "UMD01": "control",
    "UMD02": "control",
    "UMD03": "control",
    "UMD04": "control",
    "UMD05": "control",
    "UMD06": "control",
    "UMD08": "control",
    "UMD09": "control",
    "UMD10": "control",
    "UMD11": "control",
    "UMD12": "control",
    "UMD13": "control",
    "UMD14": "control",
    "UMD15": "control",
    "UMD16": "control",
    "UMD17": "control",
    "UMD18": "control",
    "UMD19": "control",
    "UMD20": "control",
    "UMD21": "control",
    "UMD22": "control",
    "UMD23": "control",
    "UMD24": "control",
    "MSUC09b": "control",
    "richardson21": "control",
    "richardson22": "control",
    "richardson23": "control",
    "richardson34": "control",
    "richardson35": "control",
    "richardson36": "control",
    "richardson37": "control",
    "richardson38": "control",
    "richardson39": "control",
    "aprocsa1554a": "aphasia",
    "aprocsa1713a": "aphasia",
    "aprocsa1731a": "aphasia",
    "aprocsa1738a": "aphasia",
    "aprocsa1833a": "aphasia",
    "aprocsa1944a": "aphasia",
    "kurland24e": "aphasia",
    "kurland25e": "aphasia",
    "MMA04a": "aphasia",
    "MMA20a": "aphasia",
    "colin08a": "aphasia",
    "colin10a2": "aphasia",
}


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--transcript-dir", type=str)
    parser.add_argument("--out-dir", type=str)
    return parser.parse_args()


def main():
    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # get a list of all speakers
    files = []
    for file in os.listdir(args.transcript_dir):
        if os.path.isfile(os.path.join(args.transcript_dir, file)) and file.endswith(
            ".cha"
        ):
            files.append(file)
    files.sort()

    print(f"{len(files)} speakers in total")

    spk_ids = []
    spk2aphasia_type = {}
    spk2aphasia_severity = {}
    with open(os.path.join(out_dir, "spk_info.txt"), "w") as f:
        f.write("spk\twab_aq\tseverity\taphasia_type\n")
        for file in files:
            spk = file.split(".cha")[0]
            assert spk not in spk_ids, spk  # spk is unique id for participants
            spk_ids.append(spk)

            path = os.path.join(args.transcript_dir, file)
            chat: pla.Reader = pla.read_chat(path)

            header = chat.headers()
            # sex = header[0]['Participants']['PAR']['sex']
            # age = header[0]['Participants']['PAR']['age']
            # age = int(age.split(';')[0])  # "year;month.day"
            aphasia_type = header[0]["Participants"]["PAR"]["group"].lower()

            wab_aq = header[0]["Participants"]["PAR"]["custom"]
            if wab_aq == "":
                # print(f'Cannot find the WAB_AQ of {spk}')
                wab_aq = "none"
                severity = "none"
            else:
                wab_aq = float(wab_aq)

                if 0 < wab_aq <= 25:
                    severity = "very_severe"
                elif 25 < wab_aq <= 50:
                    severity = "severe"
                elif 50 < wab_aq <= 75:
                    severity = "moderate"
                else:
                    severity = "mild"

            if aphasia_type == "":
                aphasia_type = spk2missing_aphasia_types[spk]

            f.write(f"{spk}\t{wab_aq}\t{severity}\t{aphasia_type}\n")

            spk2aphasia_type[spk] = aphasia_type
            spk2aphasia_severity[spk] = severity

    with open(os.path.join(out_dir, "spk2aphasia_type"), "w") as f:
        json.dump(spk2aphasia_type, f)
    with open(os.path.join(out_dir, "spk2aphasia_severity"), "w") as f:
        json.dump(spk2aphasia_severity, f)


if __name__ == "__main__":
    main()
