import json
from pprint import pprint
import spacy
from utils import timestamp_to_duration_conversion, load_dict, save_dict
from spacy_conll import init_parser
import argparse
import requests
import string


def parse_arguments():
    parser = argparse.ArgumentParser(description="Dependency parsing.")
    parser.add_argument(
        "--data-dict-path",
        type=str,
        default="",
        help="Data dict path",
    )
    parser.add_argument(
        "--use-asr-transcripts",
        action="store_true",
        help="Calculate features based on ASR transcripts",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language",
    )
    parser.add_argument(
        "--save-dict-name",
        type=str,
        help="Path to save new dict",
    )
    return parser.parse_args()


class FrenchConlluParser(object):
    def __init__(self):
        self.nlp = spacy.load("fr_dep_news_trf")
        config = {"ext_names": {"conll_pd": "pandas"}, "include_headers": True}
        # "conversion_maps": {"deprel": {"nsubj": "subj"}}}
        self.nlp.add_pipe("conll_formatter", config=config, last=True)

    def calculate_conllu_from_sentence(self, transcript: str):
        return self.nlp(transcript)._.conll_str


class EnglishConlluParser(object):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        config = {"ext_names": {"conll_pd": "pandas"}, "include_headers": True}
        # "conversion_maps": {"deprel": {"nsubj": "subj"}}}
        self.nlp.add_pipe("conll_formatter", config=config, last=True)

    def calculate_conllu_from_sentence(self, transcript: str):
        return self.nlp(transcript)._.conll_str

    def calculate_conllu(self, story_dict):
        for story in story_dict.keys():
            transcripts_dict = story_dict[story]["raw_transcriptions"]
            for transcript in transcripts_dict.keys():
                print(story, transcript)
                conllu_res = self.nlp(transcript)._.conll_str
                story_dict[story]["raw_transcriptions"][transcript][
                    "conllu"
                ] = conllu_res

        return story_dict

    def calculate_conllu_from_dict(
            self, data_dict, new_dict_format=True, use_asr_transcripts=False
    ):
        if use_asr_transcripts:
            transcript_type = "asr_prediction"
            conllu_name = "conllu-asr"
        else:
            transcript_type = "processed_transcript"
            conllu_name = "conllu"
        for speaker in data_dict.keys():
            for story in data_dict[speaker].keys():
                if story == "user_info":
                    continue
                if new_dict_format:
                    transcripts_dict = data_dict[speaker][story]["utterances"]
                else:
                    transcripts_dict = data_dict[speaker][story]["raw_transcriptions"]
                for transcript in transcripts_dict.keys():
                    if new_dict_format:
                        transcript_name = transcript
                        if (
                                transcript_type
                                not in transcripts_dict[transcript_name].keys()
                        ):
                            continue
                        transcript = transcripts_dict[transcript_name][transcript_type]
                    if transcript == "":
                        continue
                    print(story, transcript)
                    conllu_res = self.nlp(transcript)._.conll_str
                    if new_dict_format:
                        data_dict[speaker][story]["utterances"][transcript_name][
                            conllu_name
                        ] = conllu_res
                    else:
                        data_dict[speaker][story]["raw_transcriptions"][transcript][
                            conllu_name
                        ] = conllu_res
        return data_dict


def calculate_conllu(
        data_dict,
        save_dict_name,
        language,
        key_target="processed_transcript",
        conllu_name="conllu",
):
    if language == "english":
        parser = EnglishConlluParser()
    elif language == "french":
        parser = FrenchConlluParser()
    else:
        exit("Improper Language.")
    for speaker in data_dict.keys():
        for story in data_dict[speaker].keys():
            if story == "user_info":
                continue
            print(f"Processing {speaker}-{story}")
            for utterance_name in data_dict[speaker][story]["utterances"].keys():
                utterance_dict = data_dict[speaker][story]["utterances"][utterance_name]
                if key_target not in utterance_dict.keys():
                    # Some utterances may not have asr transcriptions
                    continue
                if utterance_dict[key_target] == "":
                    # Continue if transcript is empty string
                    continue
                conllu = parser.calculate_conllu_from_sentence(
                    utterance_dict[key_target]
                )
                data_dict[speaker][story]["utterances"][utterance_name][
                    conllu_name
                ] = conllu
    save_dict(data_dict, save_dict_name)


def calculate_conllus_ilsp(
        data_dict,
        key_target="processed_transcript",
        conllu_name="conllu",
        save_dict_name="planv_data.json",
):
    for speaker in data_dict.keys():
        for story in data_dict[speaker].keys():
            if story == "user_info":
                continue
            utterances, utt_names = [], []

            for utt_name in data_dict[speaker][story]["utterances"].keys():
                if (
                        key_target
                        not in data_dict[speaker][story]["utterances"][utt_name].keys()
                ):
                    continue
                t = data_dict[speaker][story]["utterances"][utt_name][key_target]
                if t == "":
                    continue

                def preprocess_text(t):
                    t = t.translate(str.maketrans("", "", string.punctuation))
                    t = t.capitalize()
                    return t + "."

                t = preprocess_text(t)
                utt_names.append(utt_name)
                utterances.append(t)
            if utterances == []:
                continue
            text = "\n\n".join(utterances)
            response = requests.post(
                "http://nlp.ilsp.gr/nws/api/", data={"text": text}, timeout=5
            ).json()
            sentences = text.split("\n\n")
            conllus = response["conllu"].split("\n\n")
            if conllus[-1] == "":
                conllus = conllus[0:-1]
            for idx, (utt_name, utterance, conllu) in enumerate(
                    zip(utt_names, utterances, conllus)
            ):
                data_dict[speaker][story]["utterances"][utt_name][conllu_name] = conllu
    save_dict(data_dict, save_dict_name)


if __name__ == "__main__":
    args = parse_arguments()
    # Asr transcripts or not: proper dictionary key names
    if args.use_asr_transcripts:
        key_target = "asr_prediction"
        conllu_name = "conllu-asr"
    else:
        key_target = "processed_transcript"
        conllu_name = "conllu"
    # Language
    if args.language == "en":
        language_name = "english"
    elif args.language == "fr":
        language_name = "french"
    elif args.language == "el":
        language_name = "greek"
    else:
        msg = "Improper language name. Process terminates."
        exit(msg)

    if args.language in ["en", "fr"]:
        calculate_conllu(
            data_dict=load_dict(args.data_dict_path),
            save_dict_name=args.save_dict_name,
            language=language_name,
            key_target=key_target,
            conllu_name=conllu_name,
        )
    else:
        # Using Greek language
        # Requests to API
        calculate_conllus_ilsp(
            data_dict=load_dict(args.data_dict_path),
            key_target=key_target,
            conllu_name=conllu_name,
            save_dict_name=args.save_dict_name,
        )
