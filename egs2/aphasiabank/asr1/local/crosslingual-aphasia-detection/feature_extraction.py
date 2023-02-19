import os
import re
import glob
import json
import spacy
import string
import argparse
import itertools
import numpy as np
from utils import (
    transcript_cleaning_decorator,
    timestamp_to_ints,
    load_dict,
    save_dict,
    get_vocabulary,
    check_words_in_vocabulary,
    create_name_value_dict,
)
from transcript_processing import TranscriptsProcessor
from conllu import parse_tree
from pprint import pprint
import argparse
from feature_names import UTTERANCE_LEVEL_FEATURE_NAMES, FEATURES_NAMES, HELPER_NAMES


def parse_arguments():
    parser = argparse.ArgumentParser(description="Feature Extraction")
    parser.add_argument(
        "--calculate-features-from-dict",
        action="store_true",
        help="Calculate features based on ASR transcripts",
    )
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


class TreeFeatureExtraction(object):
    def __init__(self):
        pass

    def calculate_tree_height(self, node):
        depth_list = []
        if node.children == []:
            # leaf node
            return 0
        for child in node.children:
            depth_list.append(self.calculate_tree_height(child))
        return 1 + max(depth_list)

    def tree_calculations(self, data):
        tree_height = 0

        try:
            tree = parse_tree(data)
        except:
            return 0
        root = tree[0]
        children = root.children

        tree_height = self.calculate_tree_height(root)
        return tree_height


class SingleLineFeatureExtractor(object):
    def __init__(self):
        self.OPEN_CLASS_TYPES = ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]
        self.PUNC = """!()-[]{};:'"\\,<>./?@#$%^&*_~"""
        self.tree_extractor = TreeFeatureExtraction()

    def calculate_features_per_utterance(self, text, conllu_str, vocabulary):
        (
            n_words,
            n_nouns,
            n_verbs,
            n_adj,
            n_cconj,
            n_adv,
            n_prepositions,
            n_close_class_words,
            n_independent,
            n_dependent,
        ) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        lemmas, tree_heights, words_in_voc = [], [], []
        # Split lines
        lines = conllu_str.splitlines()

        # Number of words
        n_words = self.calculate_number_of_words(text)

        # Words found in dictionary
        words_in_voc, _ = check_words_in_vocabulary(vocabulary, text)

        # Convert lines to list of tokens
        # Remove lines starting with #
        lists_of_tokens = self.convert_lines_to_lists_of_tokens(lines)

        for t in lists_of_tokens:
            lemma = t[2]
            lemmas.append(lemma)
            pos_tag = t[3]

            # POS counters
            if pos_tag == "NOUN" or pos_tag == "PROPN":
                n_nouns += 1
            elif pos_tag == "VERB":
                if t[-3] != "aux":
                    n_verbs += 1
                if t[-3] == "xcomp" and "VerbForm=Part" in t[5]:
                    n_verbs -= 1
                    n_adj += 1
                if t[-3] == "amod" and "VerbForm=Part" in t[5]:
                    n_verbs -= 1
                    n_adj += 1
                if t[-3] == "obj" and "VerbForm=Part" in t[5]:
                    n_verbs -= 1
                    n_nouns += 1
                if t[-3] == "nsubj" and "VerbForm=Part" in t[5]:
                    n_verbs -= 1
                    n_nouns += 1
            elif pos_tag == "ADJ":
                n_adj += 1
            elif pos_tag == "CCONJ" or pos_tag == "SCONJ":
                n_cconj += 1
            elif pos_tag == "ADV":
                n_adv += 1
            elif pos_tag == "ADP":
                n_prepositions += 1

            if pos_tag not in self.OPEN_CLASS_TYPES:
                n_close_class_words += 1

        if lists_of_tokens:
            tree_heights.append(self.tree_extractor.tree_calculations(conllu_str))
            (
                n_independent,
                n_dependent,
            ) = self.calculate_dependent_independent_sentences_per_utterance(conllu_str)
        return (
            n_words,
            n_nouns,
            n_verbs,
            n_adj,
            n_cconj,
            n_adv,
            n_prepositions,
            n_close_class_words,
            n_independent,
            n_dependent,
            lemmas,
            tree_heights,
            words_in_voc,
        )

    def calculate_number_of_words(self, utterance):
        for ele in self.PUNC:
            utterance = utterance.replace(ele, " ")
        # remove whitespaces between words and get list of words
        words = " ".join(utterance.split()).split(" ")
        return len(words)

    def convert_lines_to_lists_of_tokens(self, lines):
        token_list = []
        for line in lines:
            if (
                    ("# newpar id" in line)
                    or ("# sent_id" in line)
                    or ("# text" in line)
                    or ("# newdoc id" in line)
            ):
                continue
            if line:
                # to avoid processing empty lines
                if "PUNCT" in line:
                    continue
                tmp_line = line.split("\t")
                token_list.append(tmp_line)
        return token_list

    def calculate_dependent_independent_sentences_per_utterance(self, conllu_utterance):
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


class StoryLevelFeatureExtractor(SingleLineFeatureExtractor):
    def __init__(self, language, stack_utterances_durations, use_asr_transcripts=False):
        super().__init__()
        self.number_of_features = 24
        self.number_of_helpers = 15
        self.stack_utterances_durations = stack_utterances_durations
        self.language = language
        self.use_asr_transcripts = use_asr_transcripts
        if language == "greek":
            self.nlp = spacy.load("el_core_news_lg")
        elif language == "english":
            self.nlp = spacy.load("en_core_web_trf")
        elif language == "mandarin":
            self.nlp = spacy.load("zh_core_web_trf")
        elif language == "french":
            self.nlp = spacy.load("fr_dep_news_trf")
        self.vocabulary = get_vocabulary(self.nlp, self.language)

    def feature_extraction_story_level(
            self, story_dict, new_dict_format=False, conllu_name="conllu"
    ):
        (
            total_duration,
            n_utterances,
            n_words,
            n_nouns,
            n_verbs,
            n_adj,
            n_cconj,
            n_adv,
            n_prepositions,
            n_close_class_words,
            n_independent,
            n_dependent,
            lemmas,
            tree_heights,
            words_in_voc,
        ) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], [], [])
        if self.use_asr_transcripts:
            transcript_key = "asr_prediction"
        else:
            transcript_key = "processed_transcript"
        t_story_start, t_story_end = 0, 0
        utterances = story_dict.keys()
        n_utterances = len(utterances)
        for idx, utterance in enumerate(utterances):
            if new_dict_format:
                utterance_name = utterance
                if transcript_key not in story_dict[utterance_name].keys():
                    continue
                if story_dict[utterance_name][transcript_key] == "":
                    continue
                utterance = story_dict[utterance_name][transcript_key]
                conllu = story_dict[utterance_name][conllu_name]
                duration = story_dict[utterance_name]["duration"]
            else:
                conllu = story_dict[utterance][conllu_name]
                duration = story_dict[utterance]["duration"]
            if idx == 0:
                if not self.stack_utterances_durations:
                    if new_dict_format:
                        t_story_start, _ = timestamp_to_ints(
                            story_dict[utterance_name]["timestamp"]
                        )
                    else:
                        t_story_start, _ = timestamp_to_ints(
                            story_dict[utterance]["timestamp"]
                        )
            if self.stack_utterances_durations:
                total_duration += duration

            utterance_features = self.calculate_features_per_utterance(
                utterance, conllu, self.vocabulary
            )
            n_words += utterance_features[0]
            n_nouns += utterance_features[1]
            n_verbs += utterance_features[2]
            n_adj += utterance_features[3]
            n_cconj += utterance_features[4]
            n_adv += utterance_features[5]
            n_prepositions += utterance_features[6]
            n_close_class_words += utterance_features[7]
            n_independent += utterance_features[8]
            n_dependent += utterance_features[9]
            lemmas += utterance_features[10]
            tree_heights += utterance_features[11]
            words_in_voc += utterance_features[12]
        if not self.stack_utterances_durations:
            if new_dict_format:
                _, t_story_end = timestamp_to_ints(
                    story_dict[utterance_name]["timestamp"]
                )
            else:
                _, t_story_end = timestamp_to_ints(story_dict[utterance]["timestamp"])
            total_duration = t_story_end - t_story_start

        helpers = (
            total_duration,
            n_utterances,
            n_words,
            n_nouns,
            n_verbs,
            n_adj,
            n_cconj,
            n_adv,
            n_prepositions,
            n_close_class_words,
            n_independent,
            n_dependent,
            lemmas,
            tree_heights,
            words_in_voc,
        )

        story_features = self.calculate_features(*helpers)
        return story_features, helpers

    def calculate_features(
            self,
            total_duration,
            n_utterances,
            n_words,
            n_nouns,
            n_verbs,
            n_adj,
            n_cconj,
            n_adv,
            n_prepositions,
            n_close_class_words,
            n_independent,
            n_dependent,
            lemmas,
            tree_heights,
            words_in_voc,
    ):
        n_lemmas = len(list(set(lemmas)))
        n_open_class_words = n_nouns + n_verbs + n_adj + n_adv

        # Features
        # 1. Linguistic Productivity
        average_utterance_length = n_words / n_utterances if n_utterances != 0 else 0

        # 2. Propositional Density / Richness
        verbs_words_ratio = n_verbs / n_words if n_words != 0 else 0
        nouns_words_ratio = n_nouns / n_words if n_words != 0 else 0
        adjectives_words_ratio = n_adj / n_words if n_words != 0 else 0
        adverbs_words_ratio = n_adv / n_words if n_words != 0 else 0
        conjuctions_words_ratio = n_cconj / n_words if n_words != 0 else 0
        prepositions_words_ratio = n_prepositions / n_words if n_words != 0 else 0

        # 3. Fluency
        words_per_minute = (
            n_words / total_duration * 60 * 1000 if total_duration != 0 else 0
        )

        # 4. Syntactic Complexity
        verbs_per_utterance = n_verbs / n_utterances if n_utterances != 0 else 0

        # 5. Lexical Diversity
        unique_words_per_words = n_lemmas / n_words if n_words != 0 else 0

        # 6. Syntactic Complexity
        nouns_verbs_ratio = (n_nouns / n_verbs) if n_verbs != 0 else 0

        # 7. Word Retrieval - Gross Output
        n_words = n_words

        # 8. Syntactic Complexity
        open_close_class_ratio = (
            n_open_class_words / n_close_class_words if n_close_class_words != 0 else 0
        )

        # 9. Mean Number of Clauses per Utterance
        mean_clauses_per_utterance = (
            (n_independent + n_dependent) / n_utterances if n_utterances != 0 else 0
        )

        # 10. Mean of dependent clauses per utterance
        mean_dependent_clauses = n_dependent / n_utterances if n_utterances != 0 else 0

        # 10.1. Mean of independent clauses per utterance
        mean_dependent_clauses = (
            n_independent / n_utterances if n_utterances != 0 else 0
        )

        ## Extra 10.2 Dependent Clauses / Total Number of clauses
        dependent_all_clauses_ratio = (
            n_dependent / (n_dependent + n_independent)
            if n_dependent + n_independent != 0
            else 0
        )

        ## 11. Mean Tree Height
        mean_tree_height = (
            sum(tree_heights) / len(tree_heights) if len(tree_heights) != 0 else 0
        )

        ## 13. Max Tree Depth (equals to Tree Height)
        max_tree_depth = max(tree_heights) if tree_heights else 0

        ## Extra 1: Number of Independent Sentences (Utterance)
        n_independent = n_independent
        ## Extra 2: Number of Dependent Sentences (Utterance)
        n_dependent = n_dependent
        ## Extra 3: Number of verbs + adjectives + adverbs + prepositions + conjunctions
        ## divided by the total number of words
        propositional_density = (
            (n_verbs + n_adj + n_adv + n_prepositions + n_cconj) / n_words
            if n_words != 0
            else 0
        )
        ## Number of Words in Vocabulary per Number of Words
        words_in_vocabulary_per_words = (
            len(words_in_voc) / n_words if n_words != 0 else 0
        )
        ## Number of Unique Words in Vocabulary per Number of Words
        unique_words_in_vocabulary_per_words = (
            len(list(set(words_in_voc))) / n_words if n_words != 0 else 0
        )

        return (
            average_utterance_length,
            verbs_words_ratio,
            nouns_words_ratio,
            adjectives_words_ratio,
            adverbs_words_ratio,
            conjuctions_words_ratio,
            prepositions_words_ratio,
            words_per_minute,
            verbs_per_utterance,
            unique_words_per_words,
            nouns_verbs_ratio,
            n_words,
            open_close_class_ratio,
            mean_clauses_per_utterance,
            mean_dependent_clauses,
            mean_dependent_clauses,
            dependent_all_clauses_ratio,
            mean_tree_height,
            max_tree_depth,
            n_independent,
            n_dependent,
            propositional_density,
            words_in_vocabulary_per_words,
            unique_words_in_vocabulary_per_words,
        )


class SpeakerLevelFeatureExtractor(StoryLevelFeatureExtractor):
    def __init__(
            self,
            language,
            stack_utterances_durations=False,
            use_asr_transcripts=False,
    ):
        super().__init__(language, stack_utterances_durations, use_asr_transcripts)
        self.use_asr_transcripts = use_asr_transcripts
        self.stack_utterances_durations = stack_utterances_durations
        self.language = language
        if language == "greek":
            self.nlp = spacy.load("el_core_news_lg")
        elif language == "english":
            self.nlp = spacy.load("en_core_web_trf")
        elif language == "mandarin":
            self.nlp = spacy.load("zh_core_web_lg")
        elif language == "french":
            self.nlp = spacy.load("fr_dep_news_trf")
        self.vocabulary = get_vocabulary(self.nlp, self.language)

    def feature_extraction_speaker_level(
            self, speaker_dict, new_dict_format=False, conllu_name="conllu"
    ):
        stories = speaker_dict.keys()
        # total_features = [0] * self.number_of_features
        total_helpers = np.zeros(self.number_of_helpers - 3)
        total_lemmas, total_tree_heigths, total_words_in_voc = [], [], []
        for story in stories:
            if story in ["user_info", "features", "helpers"]:
                continue
            if new_dict_format:
                story_dict = speaker_dict[story]["utterances"]
            else:
                story_dict = speaker_dict[story]["raw_transcriptions"]
            _, helpers = self.feature_extraction_story_level(
                story_dict, new_dict_format, conllu_name=conllu_name
            )
            total_helpers = np.add(
                total_helpers, np.asarray(helpers, dtype=tuple)[0:-3]
            )
            total_lemmas += helpers[-3]
            total_tree_heigths += helpers[-2]
            total_words_in_voc += helpers[-1]
        total_helpers = (
                list(total_helpers)
                + [total_lemmas]
                + [total_tree_heigths]
                + [total_words_in_voc]
        )
        speaker_features = self.calculate_features(*total_helpers)
        return speaker_features, total_helpers

    def fetch_speaker_labels(self, speaker_dict):
        # or aq score (depends on type of classification)
        return speaker_dict["user_info"]["group"]

    def calculate_features_all_speakers(self, data_dict):
        speakers = data_dict.keys()
        X = np.zeros((len(speakers), self.number_of_features))
        labels = []

        for idx, speaker in enumerate(speakers):
            speaker_features, _ = self.feature_extraction_speaker_level(
                data_dict[speaker]
            )
            label = self.fetch_speaker_labels(data_dict[speaker])
            X[idx, :] = speaker_features
            labels.append(label)
        return X, labels

    def calculate_features_all_stories(self, data_dict):
        speakers = data_dict.keys()
        X = []
        labels = []
        speakers_list = []

        for idx_speaker, speaker in enumerate(speakers):
            for idx_story, story in enumerate(data_dict[speaker].keys()):
                if story in ["user_info", "features", "helpers"]:
                    continue
                features, _ = self.feature_extraction_story_level(
                    data_dict[speaker][story]["raw_transcriptions"]
                )
                labels.append(self.fetch_speaker_labels(data_dict[speaker]))
                X.append(features)
                speakers_list.append(speaker)
        X = np.array(X, dtype=float)
        return X, labels, speakers_list


def create_features_dict(data_dict, use_asr_transcripts, save_dict_name, language):
    if use_asr_transcripts:
        conllu_name = "conllu-asr"
    else:
        conllu_name = "conllu"

    features_dict = data_dict
    speaker_level_feature_extractor = SpeakerLevelFeatureExtractor(
        language=language,
        stack_utterances_durations=False,
        use_asr_transcripts=use_asr_transcripts,
    )
    new_dict_format = True
    for speaker in data_dict.keys():
        (
            speaker_features,
            speaker_helpers,
        ) = speaker_level_feature_extractor.feature_extraction_speaker_level(
            data_dict[speaker], new_dict_format, conllu_name=conllu_name
        )
        speaker_helpers_dict = create_name_value_dict(speaker_helpers, HELPER_NAMES)
        speaker_features_dict = create_name_value_dict(speaker_features, FEATURES_NAMES)
        features_dict[speaker]["features"] = speaker_features_dict
        features_dict[speaker]["helpers"] = speaker_helpers_dict
    save_dict(features_dict, save_dict_name)
    return features_dict


if __name__ == "__main__":
    args = parse_arguments()

    if args.language == "en":
        language_name = "english"
    elif args.language == "fr":
        language_name = "french"
    elif args.language == "el":
        language_name = "greek"
    else:
        msg = "Improper language name. Process terminates."
        exit(msg)

    if args.calculate_features_from_dict:
        create_features_dict(
            data_dict=load_dict(args.data_dict_path),
            use_asr_transcripts=args.use_asr_transcripts,
            save_dict_name=args.save_dict_name,
            language=language_name,
        )
