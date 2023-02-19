import os
import re
import glob
import string
from utils import (
    transcript_cleaning_decorator,
    timestamp_to_duration_conversion,
    save_dict,
)
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Transcript Processing.")
    parser.add_argument(
        "--create-data-aphasiabank",
        action="store_true",
        help="Create data for aphasiabank dataset",
    )
    parser.add_argument(
        "--transcripts-root-path",
        type=str,
        help="Transcripts root path",
    )
    parser.add_argument(
        "--wavs-root-path",
        type=str,
        help="Wavs root path",
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


class SingleLineProcessor(object):
    def __init__(self):
        self.punctuation = string.punctuation
        self.time_indicator_char = ""

    def remove_punctuation(self, line):
        return line.translate(str.maketrans("", "", self.punctuation))

    def fetch_timestamp(self, line):
        pattern = f"{self.time_indicator_char}(.*?){self.time_indicator_char}"
        time = re.findall(pattern, line)
        return time[0] if time else None

    def fetch_patient_metadata(self, line):
        line_split = line.split("|")
        aq_score = float(line_split[-2]) if line_split[-2] else None
        gender = line_split[4]
        aphasia_type = line_split[5]
        return aq_score, gender, aphasia_type

    def fetch_story(self, line):
        return line.replace("@G:", "").strip()


class DocumentProcessor(object):
    def __init__(self):
        self.slp = SingleLineProcessor()

    def read_cha_file(self, cha_fp):
        with open(cha_fp, "r", encoding="utf8") as fp:
            lines = fp.readlines()
        return lines

    @transcript_cleaning_decorator
    def process_cha_file(self, cha_fp, language="english", cleaning=None):
        # Read CHA file
        cha_lines = self.read_cha_file(cha_fp)

        # Initializations
        transcripts, stories, timestamps = [], [], []
        aq_score, aphasia_type, gender, story = None, None, None, None
        participant_speaking, previous_utterance = False, ""

        for idx, line in enumerate(cha_lines):
            if "@G:" in line:
                story = self.slp.fetch_story(line)
            if "@ID" in line and "PAR" in line:
                aq_score, gender, aphasia_type = self.slp.fetch_patient_metadata(line)
            if "*PAR" in line or participant_speaking:
                # if participant starts or continues speaking
                timestamp = self.slp.fetch_timestamp(line)
                previous_utterance += line
                if timestamp:
                    # if timestamp exists, it signifies that
                    # participant ended speaking
                    timestamps.append(timestamp)
                    transcripts.append(previous_utterance)
                    stories.append(story)
                    participant_speaking = False
                    previous_utterance = ""
                else:
                    # if timestamp does not exist
                    # participant continues speaking
                    participant_speaking = True
        return transcripts, stories, timestamps, aq_score, aphasia_type, gender


class TranscriptsProcessor(object):
    """
    AphasiaBank Transcript Processor Class

    Processes all transcripts (in .cha format, in the AphasiaBank style)
    Currently supports only english AB
    """

    def __init__(
            self,
            language,
            transcripts_datapath,
            group="all",
            create_data_dict=False,
            new_format=False,
    ):
        self.language = language
        self.transcripts_root_path = transcripts_datapath
        self.doc_processor = DocumentProcessor()
        self.group_list = self.find_group(group)
        self.transcripts_paths, self.groups = self.get_transcripts_paths(new_format)
        (
            self.transcripts,
            self.stories,
            self.timestamps,
            self.aq_score,
            self.aphasia_type,
            self.gender,
            self.data_dict,
        ) = self.process_transcripts(create_data_dict)

    def get_transcripts_paths(self, new_format):
        transcript_paths, group_transcripts_paths, groups = [], [], []
        for group in self.group_list:
            if not new_format:
                tmp_language = self.language.capitalize()
                tmp_group = group.capitalize()
            else:
                tmp_language = self.language
                tmp_group = group
            group_transcripts_paths = glob.glob(
                os.path.join(
                    self.transcripts_root_path,
                    # tmp_language,
                    # tmp_group,
                    "**",
                    "*.cha",
                )
            )
            groups += [group] * len(group_transcripts_paths)
            transcript_paths += group_transcripts_paths
        return transcript_paths, groups

    def find_group(self, group_argument: str):
        group_list = []
        if group_argument == "all":
            group_list = ["aphasia", "control"]
        else:
            group_list.append(group_argument)
        return group_list

    def process_transcripts(self, create_data_dict=False):
        data_dict = {}
        speakers, transcripts, stories, timestamps, aq_scores, aphasia_type, gender = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for transcript_path in self.transcripts_paths:
            (
                cha_transcripts,
                cha_stories,
                cha_timestamps,
                cha_aq,
                cha_aphasia_type,
                cha_gender,
            ) = self.doc_processor.process_cha_file(
                transcript_path, language=self.language, cleaning=None
            )
            # Alternative: cleaning="pylangacq"
            speaker = transcript_path.split("/")[-1].split(".")[0]
            group_indicated_from_path = transcript_path.split("/")[-3]
            # speakers in aphasia/control may have the same name
            # issue arises as one speaker from control group may delete data from aphasia group
            speaker = f"{speaker}-{group_indicated_from_path}"
            speakers.append(speaker)
            transcripts += cha_transcripts
            stories += cha_stories
            timestamps += cha_timestamps
            aq_scores.append(cha_aq)
            aphasia_type.append(cha_aphasia_type)
            gender.append(cha_gender)

            # Dictionary
            # Each file is a story
            if create_data_dict:
                group_indicated_from_path = transcript_path.split("/")[-3]
                group = (
                    "aphasia"
                    if group_indicated_from_path in ["Aphasia", "aphasia"]
                    else "control"
                )
                print(transcript_path, group)
                data_dict[speaker] = {"user_info": {"group": group}}
                for idx, _ in enumerate(cha_transcripts):
                    if cha_transcripts[idx] == "":
                        continue
                    cha_transcripts[idx] = " ".join(cha_transcripts[idx].split())
                    # Capitalize First letter
                    cha_transcripts[idx] = cha_transcripts[idx].capitalize()
                    # Remove punctutation
                    cha_transcripts[idx] = cha_transcripts[idx].translate(
                        str.maketrans("", "", string.punctuation)
                    )
                    # Add . at the end of the transcript
                    cha_transcripts[idx] += "."
                    if cha_stories[idx] in data_dict[speaker].keys():
                        data_dict[speaker][cha_stories[idx]]["raw_transcriptions"][
                            cha_transcripts[idx]
                        ] = {
                            "timestamp": cha_timestamps[idx],
                            "duration": timestamp_to_duration_conversion(
                                cha_timestamps[idx]
                            ),
                            "conllu": "",
                        }
                    else:
                        data_dict[speaker][cha_stories[idx]] = {
                            "raw_transcriptions": {
                                cha_transcripts[idx]: {
                                    "timestamp": cha_timestamps[idx],
                                    "duration": timestamp_to_duration_conversion(
                                        cha_timestamps[idx]
                                    ),
                                    "conllu": "",
                                }
                            }
                        }
        return (
            transcripts,
            stories,
            timestamps,
            aq_scores,
            aphasia_type,
            gender,
            data_dict,
        )

    def story_level_transcription(self):
        pass


class TranscriptsProcessorAphasiaBank(TranscriptsProcessor):
    def __init__(
            self,
            language,
            transcripts_datapath,
            wavs_root_path,
            create_data_dict,
    ):
        self.wavs_root_path = wavs_root_path
        super().__init__(
            language=language,
            transcripts_datapath=transcripts_datapath,
            create_data_dict=create_data_dict,
        )
        print(self.wavs_root_path)

    def process_transcripts(self, create_data_dict=False):
        data_dict = {}
        speakers, transcripts, stories, timestamps, aq_scores, aphasia_type, gender = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        transcriptions_with_no_audios = []
        for transcript_path in self.transcripts_paths:
            (
                cha_transcripts,
                cha_stories,
                cha_timestamps,
                cha_aq,
                cha_aphasia_type,
                cha_gender,
            ) = self.doc_processor.process_cha_file(
                transcript_path, language=self.language, cleaning=None
            )
            # Alternative: cleaning="pylangacq"
            speaker = transcript_path.split("/")[-1].split(".")[0]
            group_indicated_from_path = transcript_path.split("/")[-3]
            # speakers in aphasia/control may have the same name
            # issue arises as one speaker from control group may delete data from aphasia group
            speaker = f"{speaker}-{group_indicated_from_path}"
            speakers.append(speaker)
            transcripts += cha_transcripts
            stories += cha_stories
            timestamps += cha_timestamps
            aq_scores.append(cha_aq)
            aphasia_type.append(cha_aphasia_type)
            gender.append(cha_gender)

            # Dictionary
            # Each file is a story
            if create_data_dict:
                group_indicated_from_path = transcript_path.split("/")[-3]
                group = (
                    "aphasia"
                    if group_indicated_from_path in ["Aphasia", "aphasia"]
                    else "control"
                )
                # tmp = transcript_path.split(".")[0].split("/")
                # fp = f"{self.wavs_root_path}/{'/'.join(tmp[-3::])}"
                # for extension in ["wav", "mp4", "mp3"]:
                #     wav_file = f"{fp}.{extension}"
                #     if os.path.isfile(wav_file):
                #         break
                # if not os.path.isfile(wav_file):
                #     # print("Cannot find audio file", transcript_path)
                #     transcriptions_with_no_audios.append(transcript_path)
                #     continue
                data_dict[speaker] = {
                    "user_info": {
                        "group": group,
                        "transcript_path": transcript_path,
                        # "audio_path": wav_file,
                        "audio_path": '',
                    }
                }
                for idx, _ in enumerate(cha_transcripts):
                    if cha_transcripts[idx] == "":
                        continue
                    cha_transcripts[idx] = " ".join(cha_transcripts[idx].split())
                    # Capitalize First letter
                    cha_transcripts[idx] = cha_transcripts[idx].capitalize()
                    # Remove punctutation
                    cha_transcripts[idx] = cha_transcripts[idx].translate(
                        str.maketrans("", "", string.punctuation)
                    )
                    # Add . at the end of the transcript
                    cha_transcripts[idx] += "."
                    if cha_stories[idx] in data_dict[speaker].keys():
                        speaker_utterance_idx = (
                                len(data_dict[speaker][cha_stories[idx]]["utterances"]) + 1
                        )
                        data_dict[speaker][cha_stories[idx]]["utterances"][
                            f"{speaker}-{cha_stories[idx]}-{speaker_utterance_idx}"
                        ] = {
                            "processed_transcript": cha_transcripts[idx],
                            "timestamp": cha_timestamps[idx],
                            "duration": timestamp_to_duration_conversion(
                                cha_timestamps[idx]
                            ),
                            "conllu": "",
                        }
                    else:
                        data_dict[speaker][cha_stories[idx]] = {
                            "utterances": {
                                f"{speaker}-{cha_stories[idx]}-1": {
                                    "processed_transcript": cha_transcripts[idx],
                                    "timestamp": cha_timestamps[idx],
                                    "duration": timestamp_to_duration_conversion(
                                        cha_timestamps[idx]
                                    ),
                                    "conllu": "",
                                }
                            }
                        }
        print("Transcriptions with no audios", transcriptions_with_no_audios)
        return (
            transcripts,
            stories,
            timestamps,
            aq_scores,
            aphasia_type,
            gender,
            data_dict,
        )


def create_initial_data_dict_aphasiabank(
        language, transcripts_root_path, wavs_root_path, save_dict_name
):
    tp = TranscriptsProcessorAphasiaBank(
        language=language,
        transcripts_datapath=transcripts_root_path,
        wavs_root_path=wavs_root_path,
        create_data_dict=True,
    )
    data_dict = tp.data_dict
    save_dict(data_dict, save_dict_name)


if __name__ == "__main__":
    args = parse_arguments()
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

    if args.create_data_aphasiabank:
        create_initial_data_dict_aphasiabank(
            language=language_name,
            transcripts_root_path=args.transcripts_root_path,
            wavs_root_path=args.wavs_root_path,
            save_dict_name=args.save_dict_name,
        )
