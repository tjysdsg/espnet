import string
import re
import os
import json
import logging
from typing import List, Tuple
import numpy as np
from pathlib import Path
import shutil
import pylangacq
from clean_utterance import clean_utterance

# Available narrative stories
narrative_stories = [
    "Speech",
    "Stroke",
    "Important_Event",
    "Window",
    "Umbrella",
    "Cat",
    "Cinderella_Intro",
    "Cinderella",
    "Sandwich",
    "Cinderella_intro",
    "Flood",
    "Sandwich_Intro",
    "Sandwich_Picture",
    "Important_Event_Substitution",
    "Sandwich_other",
    "CAT",
    "Sandwich_Other",
    "recovery",
    "Recovery",
    "Illness_or_Injury",
]


def calculate_dependent_independent_sentences_per_utterance_greek(conllu_utterance):
    n_dependent_utterance, n_independent_utterance = 0, 0
    tokens = conllu_utterance.split("\n")
    tokens = [token for token in tokens if token and token[0] != "#"]
    prev_sentence = ""
    for token in tokens:
        tmp = token.split("\t")
        pos_tag = tmp[3]
        rel = tmp[-3]
        if pos_tag != "VERB":
            continue
        if rel == "xcomp" and "VerbForm=Part" in tmp[5]:
            continue
        if rel == "amod" and "VerbForm=Part" in tmp[5]:
            continue
        if rel == "obj" and "VerbForm=Part" in tmp[5]:
            continue
        if rel == "nsubj" and "VerbForm=Part" in tmp[5]:
            continue
        if rel == "cop":
            goto_id = tmp[-4]
            goto_token = tokens[int(goto_id) - 1].split("\t")
            if int(goto_token[0]) != int(goto_id):
                ind = int(goto_id) - int(goto_token[0])
                goto_token = tokens[int(goto_id) - 1 + ind].split("\t")
            if goto_token[3] == "ADJ":
                goto_rel = goto_token[-3]
                if goto_rel in [
                    "advcl",
                    "acl:relcl",
                    "ccomp",
                    "xcomp",
                    "acl",
                    "csubj",
                    "csubj:pass",
                ]:
                    n_dependent_utterance += 1
                    prev_sentence = "dependent"
                if goto_rel in ["root"]:
                    n_independent_utterance += 1
                    prev_sentence = "independent"
                continue
        if rel == "parataxis":
            goto_id = tmp[-4]
            goto_token = tokens[int(goto_id) - 1].split("\t")
            if goto_token[3] == "VERB":
                goto_rel = goto_token[-3]
                if goto_rel in [
                    "advcl",
                    "acl:relcl",
                    "ccomp",
                    "xcomp",
                    "acl",
                    "csubj",
                    "csubj:pass",
                ]:
                    n_dependent_utterance += 1
                    prev_sentence = "dependent"
                if goto_rel in ["root"]:
                    n_independent_utterance += 1
                    prev_sentence = "independent"
                continue
        if rel == "conj":
            if prev_sentence == "independent":
                n_independent_utterance += 1
            elif prev_sentence == "dependent":
                n_dependent_utterance += 1
            else:
                # print("conj sentence with no previous dependent/independent sentence")
                pass
        if rel in [
            "advcl",
            "acl:relcl",
            "ccomp",
            "xcomp",
            "acl",
            "csubj",
            "csubj:pass",
        ]:
            n_dependent_utterance += 1
            prev_sentence = "dependent"
        if rel in ["root"]:
            n_independent_utterance += 1
            prev_sentence = "independent"
    return n_independent_utterance, n_dependent_utterance


def timestamp_to_duration_conversion(timestamp: str):
    t_start_end_list = timestamp.split("_")
    return float(t_start_end_list[1]) - float(t_start_end_list[0])


def timestamp_to_ints(timestamp: str):
    t_start_end_list = timestamp.split("_")
    return float(t_start_end_list[0]), float(t_start_end_list[1])


def transcript_cleaning_decorator(process_cha_files):
    def cleaning_wrapper(_, cha_file, language, cleaning):
        transcripts, stories, timestamps, aq, aphasia_type, gender = process_cha_files(
            _, cha_file, language, cleaning
        )
        result_transcript = []
        for transcript in transcripts:
            if cleaning != "pylangacq":
                # remove *par, tabls and new lines
                tmp = (
                    transcript.replace("*PAR:", "").replace("\n", "").replace("\t", "")
                )
                # delete all words that start with @
                tmp = re.sub("@\\w+ *", "", tmp)
                # delete all words that start with &
                # tmp = re.sub("[\\w+ *", "", tmp)
                tmp = " ".join(filter(lambda x: x[0] != "+", tmp.split()))
                tmp = " ".join(filter(lambda x: x[0] != "&", tmp.split()))
                # tmp = " ".join(filter(lambda x:x[0]!='<', tmp.split()))
                # tmp = re.sub(r'\.+\','',tmp)
                tmp = re.sub(r" [\(\[].*?[\)\]]", "", tmp)  # [], ()
                tmp = re.sub(r"[\<\[].*?[\>\]]", "", tmp)  # <>
                tmp = re.sub(r"[\\[].*?[\\]]", "", tmp)  # <>
                tmp = re.sub(r"\&.+", "", tmp)
                tmp = tmp.replace("‡", "").replace("xxx", "").replace("„", "")
                tmp = bytes(tmp, "utf-8").decode("utf-8", "ignore")
                tmp = tmp.replace("_", " ")  # words you_know -> you know
                tmp = tmp.translate(str.maketrans("", "", string.punctuation))
                tmp = re.sub(" +", " ", tmp)
                tmp = tmp.lstrip()
                if language != "mandarin":
                    tmp = tmp.encode("ascii", "ignore").decode()
            else:
                tmp = (
                    transcript.replace("*PAR:", "").replace("\n", "").replace("\t", "")
                )
                tmp = clean_utterance(tmp)
            result_transcript.append(tmp)

        return result_transcript, stories, timestamps, aq, aphasia_type, gender

    return cleaning_wrapper


def fetch_value_inside_token(line: str, token: str) -> str:
    res = re.search(f"<{token}>(.+?)</{token}>", line)
    if res:
        return res.group(1)
    return ""


def fetch_transcriptions_from_elan_file(
        fp: str, special_tier: str = 'TIER_ID="Trascription - Patient"'
) -> List[str]:
    prev_line = ""
    correct_tier = False
    sentences = []
    t_start, t_end, durations = [], [], []
    time_dict = {}
    with open(fp, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "TIME_VALUE" in line:
            segs = line.split(" ")
            time_slot_id, time_value = None, None
            for seg in segs:
                if "TIME_SLOT_ID" in seg:
                    time_slot_id = seg.split('"')[1]
                if "TIME_VALUE" in seg:
                    time_value = int(seg.split('"')[1])
            time_dict[time_slot_id] = time_value
        if special_tier in line:
            prev_line = line
            correct_tier = True
            continue
        if "CVE_REF" in prev_line:
            prev_line = line
            continue
        if not correct_tier:
            continue
        if "TIER" in line and correct_tier:
            correct_tier = False
            continue
        res = fetch_value_inside_token(line, "ANNOTATION_VALUE")
        # Fetch timestamp
        if res:
            sentences.append(res)
            if prev_line:
                if "ALIGNABLE_ANNOTATION" in prev_line:
                    prev_segs = prev_line.split(" ")
                    for prev_seg in prev_segs:
                        if "TIME_SLOT_REF1" in prev_seg:
                            start = time_dict[prev_seg.split('"')[1]]
                        if "TIME_SLOT_REF2" in prev_seg:
                            end = time_dict[prev_seg.split('"')[1]]
                    durations.append(int(end) - int(start))
                    t_start.append(int(start))
                    t_end.append(int(end))
        prev_line = line
    return sentences, t_start, t_end, durations


def reformat_sentence(transcriptions):
    # Sentences should begin with Uppercase letter and end with .
    # And no gaps
    new_transcriptions = []
    for transcript in transcriptions:
        # Remove whitespaces
        transcript = " ".join(transcript.split())
        # Capitalize First letter
        transcript = transcript.capitalize()
        # Remove punctutation
        transcript = transcript.translate(str.maketrans("", "", string.punctuation))
        # Add . at the end of the transcript
        transcript += "."
        new_transcriptions.append(transcript)
    return new_transcriptions


def get_user_and_stories_from_eaf_files(
        eaf_files: List[str],
) -> List[Tuple[str, str, str]]:
    res = []
    for ff in eaf_files:
        fn = ff.split("/")[-1]
        if "-" in fn:
            fn = fn.replace("-", "_")
        if " " in fn:
            fn = fn.replace(" ", "_")
        tmp = fn.split("_")

        user = tmp[0]
        story = "_".join(tmp[1::])
        res.append((user, story, ff))
    return res


def save_dict(data_dict, name):
    with open(name, "w", encoding="utf8") as summary_json:
        summary_json.write(json.dumps(data_dict, indent=4, ensure_ascii=False))


def load_dict(name):
    with open(name, "rb") as json_file:
        data_dict = json.load(json_file)
    return data_dict


# https://github.com/huggingface/transformers/issues/3050
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{"|".join(prefices)})')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def save_numpy(numpy_fn, array):
    with open(numpy_fn, "wb") as f:
        np.save(f, array)


def load_numpy(numpy_fn):
    with open(numpy_fn, "rb") as f:
        array = np.load(f)
    return array


ignore_speakers = [
    "wright18a-Control",
    "MMA03a-Aphasia",
    "wright14a-Control",
    "wright13a-Control",
    "wright28a-Control",
    "wright15a-Control",
    "MMA21a-Aphasia",
    "wright21a-Control",
    "wright31a-Control",
    "wright49a-Control",
    "wright50a-Control",
    "wright27a-Control",
    "wright30a-Control",
    "MMA20a-Aphasia",
    "adler07a-Aphasia",
    "wright48a-Control",
]


def get_statistics_for_splits(idx, y_train, y_test):
    (train_unique, train_counts) = np.unique(y_train, return_counts=True)
    (test_unique, test_counts) = np.unique(y_test, return_counts=True)
    stats_dict = {
        "train": {
            "counts": {
                "aphasia": train_counts[1],
                "control": train_counts[0],
            },
            "percentages": {
                "aphasia": train_counts[1] / len(y_train) * 100,
                "control": train_counts[0] / len(y_train) * 100,
            },
        },
        "test": {
            "counts": {
                "aphasia": test_counts[1],
                "control": test_counts[0],
            },
            "percentages": {
                "aphasia": test_counts[1] / len(y_test) * 100,
                "control": test_counts[0] / len(y_test) * 100,
            },
        },
    }

    return stats_dict


def create_name_value_dict(values, names):
    return {names[idx]: values[idx] for idx in range(len(names))}


def check_words_in_vocabulary(vocabulary, text):
    # Remove punctuation and lowercase string
    text = text.translate(str.maketrans("", "", string.punctuation)).lower()
    counter = 0
    words_in_vocabulary = []
    for word in text.split():
        if word in vocabulary:
            counter += 1
            words_in_vocabulary.append(word)
    return words_in_vocabulary, counter


def get_vocabulary(nlp, language):
    return set(nlp.vocab.strings)


def reformat_goutsos_directory(path):
    aphasia_path = os.path.join(path, "aphasia")
    cha_files = [
        f
        for f in os.listdir(aphasia_path)
        if os.path.isfile(os.path.join(aphasia_path, f))
    ]
    for cha_file in cha_files:
        speaker_id = cha_file[0:2]
        speaker_folder = os.path.join(aphasia_path, speaker_id)
        Path(speaker_folder).mkdir(parents=True, exist_ok=True)
        source_file = os.path.join(aphasia_path, cha_file)
        destination_file = os.path.join(speaker_folder, cha_file)
        shutil.move(source_file, destination_file)


def pylangacq_transcript_parsing(fn):
    chat_utterances_list, t_start_list, t_end_list = [], [], []
    reader = pylangacq.read_chat(fn)
    utterances_obj_list = reader.utterances(participants="PAR")
    for utterance_obj in utterances_obj_list:
        words = [token_obj_list.word for token_obj_list in utterance_obj.tokens]
        utterance = " ".join(words)
        time_marks = utterance_obj.time_marks
        t_start, t_end = time_marks if time_marks else None, None
        # Update Results list
        chat_utterances_list.append(utterance)
        t_start_list.append(t_start)
        t_end_list.append(t_end)
    return chat_utterances_list, t_start_list, t_end_list


def normalize_text(text):
    text = text.replace("<s>", "")
    text = text.replace("</s>", "")
    text = re.sub(r"</?\[\d+>", "", text)
    text = text.replace("<unk>", "")
    text = text.replace("<pad>", "")
    text = text.lower()
    text = " ".join(text.split())
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


if __name__ == "__main__":
    pass
