import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from utils import load_dict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiments Argument Parser")
    parser.add_argument(
        "--data-dict-path-source",
        type=str,
        help="Data dict path source",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        help="Classification model",
    )
    return parser.parse_args()


def get_dataset(data_dict):
    from config import spk2aphasia_label, train_spks, test_spks

    x_train, x_test, y_train, y_test = [], [], [], []
    for speaker in data_dict.keys():
        spk = speaker.split('-')[0]  # <spk>-transcripts
        label = 1 if spk2aphasia_label[spk] == 'APH' else 0

        features = list(data_dict[speaker]["features"].values())
        if spk in train_spks:
            x_train.append(features)
            y_train.append(label)
        elif spk in test_spks:
            x_test.append(features)
            y_test.append(label)
        else:  # skip validation
            continue

    return np.asarray(x_train), np.asarray(x_test), np.asarray(y_train), np.asarray(y_test)


def loso_english_experiment(data_dict):
    x_train, x_test, y_train, y_test = get_dataset(data_dict)

    nan_ids = np.any(np.isnan(x_train), axis=0)
    x_train = x_train[:, ~nan_ids]
    nan_ids = np.any(np.isnan(x_test), axis=0)
    x_test = x_test[:, ~nan_ids]

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    for model_name in ["xgboost", "svm", "tree", "forest"]:
        if model_name == "svm":
            clf = make_pipeline(
                StandardScaler(), SVC(gamma="auto", kernel="linear")
            )
        elif model_name == "xgboost":
            clf = make_pipeline(StandardScaler(), XGBClassifier(eta=0.2))
        elif model_name == "tree":
            clf = make_pipeline(StandardScaler(), tree.DecisionTreeClassifier())
        elif model_name == "forest":
            clf = make_pipeline(StandardScaler(), RandomForestClassifier())
        else:
            assert False

        clf.fit(x_train, y_train)
        # y_preds = clf.predict(x_test)
        acc = clf.score(x_test, y_test)

        print(f"Model: {model_name} - Accuracy: {acc}")


if __name__ == "__main__":
    args = parse_arguments()
    loso_english_experiment(data_dict=load_dict(args.data_dict_path_source))
