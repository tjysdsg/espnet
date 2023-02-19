import argparse
from datetime import datetime
from datasets import load_metric
import librosa
import os
from utils import (
    load_dict,
    save_dict,
    normalize_text,
    timestamp_to_ints,
    ignore_speakers,
)
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
from pyctcdecode import build_ctcdecoder
import numpy as np
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Automatic Speech Recognition")

    parser.add_argument(
        "--data-dict-path",
        type=str,
        default="",
        help="Data dict path",
    )
    parser.add_argument(
        "--asr-evaluation",
        action="store_true",
        help="Evaluate ASR model",
    )
    parser.add_argument(
        "--asr-logits-extraction",
        action="store_true",
        help="Extract ASR logits",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language",
    )
    parser.add_argument(
        "--asr-logits-path",
        type=str,
        help="Path to save asr logits",
    )
    parser.add_argument(
        "--asr-decoding",
        action="store_true",
        help="ASR decoding process",
    )
    parser.add_argument(
        "--kenlm-model-path",
        type=str,
        help="Path to kenlm language model",
    )
    parser.add_argument(
        "--save-dict-name",
        type=str,
        help="Path to save new dict",
    )
    parser.add_argument(
        "--add-language-model",
        action="store_true",
        help="Add language model at decoding",
    )
    return parser.parse_args()


def asr_evaluation(
        data_dict,
        target_key="processed_transcript",
        prediction_key="asr_prediction",
        per_category=True,
):
    wer = load_metric("wer")
    cer = load_metric("cer")

    if per_category:
        ref_aphasia, pred_aphasia = [], []
        ref_control, pred_control = [], []
    references, predictions = [], []
    for speaker in data_dict.keys():
        for story in data_dict[speaker].keys():
            if story == "user_info":
                continue
            for utt_name in data_dict[speaker][story]["utterances"]:
                utt_dict = data_dict[speaker][story]["utterances"][utt_name]
                if prediction_key not in utt_dict.keys():
                    # AphasiaBank: some transcripts are missing
                    # print("ASR transcription not found")
                    continue
                reference = normalize_text(utt_dict[target_key])
                prediction = normalize_text(utt_dict[prediction_key])
                if reference == "":
                    continue
                references.append(reference)
                predictions.append(prediction)
                if per_category:
                    if data_dict[speaker]["user_info"]["group"] in [
                        "Aphasia",
                        "aphasia",
                    ]:
                        ref_aphasia.append(reference)
                        pred_aphasia.append(prediction)
                    else:
                        ref_control.append(reference)
                        pred_control.append(prediction)
    if per_category:
        wer_aphasia = wer.compute(references=ref_aphasia, predictions=pred_aphasia)
        cer_aphasia = cer.compute(references=ref_aphasia, predictions=pred_aphasia)
        wer_control = wer.compute(references=ref_control, predictions=pred_control)
        cer_control = cer.compute(references=ref_control, predictions=pred_control)
        print(f"Aphasia: WER={wer_aphasia} CER={cer_aphasia}")
        print(f"Control: WER={wer_control} CER={cer_control}")
    wer_result = wer.compute(references=references, predictions=predictions)
    cer_result = cer.compute(references=references, predictions=predictions)
    print(f"Total: WER={wer_result} CER={cer_result}")
    return wer_result, cer_result


def aphasiabank_asr_logits(
        data_dict,
        save_logits_path,
        language,
        batch_size=4,
):
    # Create logits directory if inexistent
    Path(save_logits_path).mkdir(parents=True, exist_ok=True)

    if language == "en":
        MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    elif language == "fr":
        MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
    # elif language == "el":
    #     MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-greek"
    else:
        msg = "Improper language name. Process terminates."
        exit(msg)

    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to("cuda")

    print("Speaker processing started")
    for idx, speaker in enumerate(data_dict.keys()):
        print(f"Speaker {speaker} under processing")
        timeit_start = datetime.now()
        if "converted_audio_path" in data_dict[speaker]["user_info"].keys():
            audio_path = data_dict[speaker]["user_info"]["converted_audio_path"]
        else:
            audio_path = data_dict[speaker]["user_info"]["audio_path"]
        for story in data_dict[speaker].keys():
            if story in ["user_info", "features", "helpers"]:
                continue
            speech_arrays = []
            utterance_names = []

            for utt_name in data_dict[speaker][story]["utterances"]:
                utt_dict = data_dict[speaker][story]["utterances"][utt_name]
                if "asr_prediction" in utt_dict.keys():
                    continue
                t_start, t_end = timestamp_to_ints(utt_dict["timestamp"])
                offset = t_start / 1000
                duration = (t_end - t_start) / 1000
                if duration > 100:
                    print(utt_name)
                    continue
                speech_array, sampling_rate = librosa.load(
                    audio_path, sr=16_000, offset=offset, duration=duration
                )
                if speech_array.shape[0] == 0:
                    continue
                utterance_names.append(utt_name)
                speech_arrays.append(speech_array)
            if utterance_names == []:
                continue
            batches = [
                speech_arrays[i: i + batch_size]
                for i in range(0, len(speech_arrays), batch_size)
            ]
            batches_names = [
                utterance_names[i: i + batch_size]
                for i in range(0, len(utterance_names), batch_size)
            ]
            total_predicted_sentences = []
            for batch, batch_names in zip(batches, batches_names):
                try:
                    inputs = processor(
                        batch,
                        sampling_rate=16_000,
                        return_tensors="pt",
                        padding=True,
                    ).to("cuda")
                    with torch.no_grad():
                        logits = model(
                            inputs.input_values, attention_mask=inputs.attention_mask
                        ).logits
                except:
                    # Resolve issue > Move calculation to cpu
                    inputs = processor(
                        batch,
                        sampling_rate=16_000,
                        return_tensors="pt",
                        padding=True,
                    )
                    model.to("cpu")
                    with torch.no_grad():
                        logits = model(
                            inputs.input_values,
                            attention_mask=inputs.attention_mask,
                        ).logits
                    model.to("cuda")

                # For each audio file in batch
                # Save its logits
                for j, b_name in enumerate(batch_names):
                    tmp = logits[j].cpu().numpy()
                    np.save(f"{save_logits_path}/{b_name}.npy", tmp)

        timeit_end = datetime.now()
        print(
            f"Speaker {idx}/{len(data_dict)} \t Took time: {timeit_end - timeit_start}"
        )


def get_kenlm_decoder(vocab_dict, kenlm_model_path, to_lower=True):
    if to_lower:
        sorted_vocab_dict = {
            k.lower(): v
            for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])
        }
    else:
        sorted_vocab_dict = {
            k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])
        }
    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()),
        kenlm_model_path=kenlm_model_path,
    )
    return decoder


def aphasiabank_decoding_process(
        data_dict,
        kenlm_model_path,
        save_dict_name,
        add_language_model,
        logits_path,
        language,
        beam_width=10,
        num_processes=10,
):
    if language == "en":
        MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    elif language == "fr":
        MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
    # elif language == "el":
    #     MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-greek"
    else:
        msg = "Improper language name. Process terminates."
        exit(msg)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

    if add_language_model:
        vocab_dict = processor.tokenizer.get_vocab()
        kenlm_decoder = get_kenlm_decoder(
            vocab_dict=vocab_dict, kenlm_model_path=kenlm_model_path
        )
        processor_with_lm = Wav2Vec2ProcessorWithLM(
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            decoder=kenlm_decoder,
        )

    for idx, speaker in enumerate(data_dict.keys()):
        if speaker in ignore_speakers:
            continue
        print(f"Speaker {speaker} under processing")
        audio_path = data_dict[speaker]["user_info"]["converted_audio_path"]
        timeit_start = datetime.now()
        for story in data_dict[speaker].keys():
            if story in ["user_info", "features", "helpers"]:
                continue
            speech_arrays = []
            utterance_names = []
            total_predicted_sentences = []

            tmp_logits_list, dimensions = [], []
            max_dim = 0
            for idx2, utt_name in enumerate(data_dict[speaker][story]["utterances"]):
                if not os.path.isfile(f"{logits_path}/{utt_name}.npy"):
                    # If numpy array does not exist continue process
                    continue
                tmp_logit = np.load(f"{logits_path}/{utt_name}.npy")
                tmp_logits_list.append(tmp_logit)
                utterance_names.append(utt_name)
                dimensions.append(tmp_logit.shape[0])

            if dimensions == []:
                print(f"Issue with story {story}: no dimensions")
                continue
            # zero padding
            max_dim = max(dimensions)
            padded_logits_list = []
            for l in tmp_logits_list:
                extra_pad = np.zeros((max_dim - l.shape[0], l.shape[1]))
                l = np.append(l, extra_pad, axis=0)
                padded_logits_list.append(l)
            logits = np.array(padded_logits_list)

            if add_language_model:
                # Decoding using language models (beam search etc)
                predicted_sentences = processor_with_lm.batch_decode(
                    logits,
                    beam_width=beam_width,
                    num_processes=num_processes,
                ).text
            else:
                # Greedy Decoding
                predicted_ids = torch.argmax(
                    torch.Tensor(logits), dim=-1
                )  # probably needs other dimensions
                predicted_sentences = processor.batch_decode(predicted_ids)
            for idx3, utt_name in enumerate(utterance_names):
                data_dict[speaker][story]["utterances"][utt_name][
                    "asr_prediction"
                ] = predicted_sentences[idx3]

        timeit_end = datetime.now()
        time_took = timeit_end - timeit_start
        print(f"Speaker {idx}/{len(data_dict)} \t Took time: {time_took}")
        save_dict(data_dict, save_dict_name)
        exit()


if __name__ == "__main__":
    args = parse_arguments()
    if args.asr_evaluation:
        asr_evaluation(data_dict=load_dict(args.data_dict_path))

    if args.asr_logits_extraction:
        aphasiabank_asr_logits(
            data_dict=load_dict(args.data_dict_path),
            save_logits_path=args.asr_logits_path,
            language=args.language,
            batch_size=4,
        )

    if args.asr_decoding:
        aphasiabank_decoding_process(
            data_dict=load_dict(args.data_dict_path),
            kenlm_model_path=args.kenlm_model_path,
            save_dict_name=args.save_dict_name,
            add_language_model=args.add_language_model,
            logits_path=args.asr_logits_path,
            language=args.language,
            beam_width=10,
            num_processes=10,
        )
