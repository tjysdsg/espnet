import argparse
import os
from speechocean762 import load_phone_symbol_table
from ase_score import eval_scoring
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, global_step_from_engine
from scoring_model import N_PHONES, load_data, data2array
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', metavar='ACTION', choices=['train', 'test'], help='train or test')
    parser.add_argument('hyp', metavar='HYP', type=str, help='Hypothesis file')
    parser.add_argument('ref', metavar='REF', type=str, help='Reference file')
    parser.add_argument('--scores', type=str, help='Path to utt2scores')
    parser.add_argument('--phone-table', type=str, help='Path to phones-pure.txt')
    parser.add_argument('--model-dir', type=str, default='tmp/scoring_model', help='Where to save the model')
    parser.add_argument('--output-dir', type=str, default='tmp/scoring_model', help='Output directory')
    parser.add_argument('--use-probs', action='store_true', help='Whether HYP contains tokens or probability matrices')
    parser.add_argument('--include-1s', action='store_true', help='Exclude score-1 samples from training and testing')
    return parser.parse_args()


class ScoringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(N_PHONES * 2, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            # ===
            nn.Linear(128, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            # ===
            nn.Linear(256, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class ProbMatrixDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def calc_and_print_metrics(preds: np.ndarray, y: np.ndarray):
    pcc, mse = eval_scoring(preds, y)
    print(f'Pearson Correlation Coefficient: {pcc:.4f}')
    print(f'MSE: {mse:.4f}')
    return preds, mse, pcc


def train_network(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, output_dir: str):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    criterion = nn.MSELoss()
    trainer = create_supervised_trainer(model, optimizer, criterion)

    val_metrics = {"loss": Loss(criterion)}
    evaluator = create_supervised_evaluator(model, metrics=val_metrics)

    # save best
    score_function = Checkpoint.get_default_score_fn("loss", score_sign=-1)
    to_save = {'model': model}
    handler = Checkpoint(
        to_save,
        DiskSaver(output_dir, create_dir=True, require_empty=False),
        score_function=score_function,
        score_name="loss",
        filename_pattern='{name}.{ext}',
        global_step_transform=global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)

    # early stopping
    handler = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    # log training acc per epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(f"Epoch[{trainer.state.epoch}]: train_loss: {metrics['loss']:.2f}")

    # log eval acc per epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(f"Epoch[{trainer.state.epoch}]: val_loss: {metrics['loss']:.2f}")

    trainer.run(train_loader, max_epochs=200)


def main():
    args = get_args()
    scaler_path = os.path.join(args.model_dir, 'scaler.pkl')
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    ph2int, int2ph = load_phone_symbol_table(args.phone_table)
    data = load_data(args.hyp, args.ref, args.scores, use_probs=True, int2ph=int2ph)

    # remove duplicates from data
    if args.use_probs and args.action == 'train':
        data = list(set(data))

    # exclude score-1 samples
    if not args.include_1s:
        new_data = []
        for d in data:
            if d.score != 1:
                new_data.append(d)
        data = new_data

    # save samples to file
    of = open(os.path.join(args.output_dir, 'samples'), 'w')
    for sam in data:
        of.write(f'CPL: {sam.cpl}\tPPL: {sam.ppl}\tScore: {sam.score}\n')
    of.close()

    X, Y = data2array(data, ph2int, True)
    X = np.asarray(X, dtype='float32')
    Y = np.asarray(Y, dtype='float32')
    Y = Y.reshape(-1, 1)
    if args.action == 'train':
        # scaler = StandardScaler()
        # scaler.fit_transform(X)

        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
        train_loader = DataLoader(ProbMatrixDataset(x_train, y_train), batch_size=64, shuffle=True)
        val_loader = DataLoader(ProbMatrixDataset(x_val, y_val), batch_size=64, shuffle=True)

        mdl = ScoringModel()
        train_network(mdl, train_loader, val_loader, args.model_dir)

        # pickle.dump(scaler, open(scaler_path, 'wb'))
    elif args.action == 'test':
        # scaler = pickle.load(open(scaler_path, 'rb'))
        # scaler.transform(X)

        test_loader = DataLoader(ProbMatrixDataset(X, Y), batch_size=64, shuffle=True)
        model_path = os.path.join(args.model_dir, 'model.pt')
        mdl = ScoringModel()
        mdl.load_state_dict(torch.load(model_path))

        pred_all = []
        y_all = []
        for x, y in test_loader:
            pred = mdl(x)
            pred_all.append(pred.detach().cpu().numpy().ravel())
            y_all.append(y.detach().cpu().numpy().ravel())

        pred_all = np.concatenate(pred_all)
        y_all = np.concatenate(y_all)
        calc_and_print_metrics(pred_all, y_all)
    else:
        assert False


if __name__ == '__main__':
    main()
