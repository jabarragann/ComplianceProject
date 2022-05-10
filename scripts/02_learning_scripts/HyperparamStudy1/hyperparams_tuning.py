from dataclasses import dataclass
from pathlib import Path
import pickle

# Torch
import optuna
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# Custom
# from kincalib.Learning.Dataset import JointsDataset, JointsRawDataset, Normalizer
from kincalib.Learning.Dataset2 import JointsDataset1, Normalizer
from kincalib.Learning.HyperparameterTuner import OptuneStudyAbstract
from kincalib.Learning.Trainer import TrainRegressionNet
from kincalib.Learning.Models import MLP, CustomMLP


@dataclass
class RegressionStudy1(OptuneStudyAbstract):
    epochs: int = 0
    loss_metric: nn.Module = None
    train_dataset: torch.utils.data.Dataset = None
    valid_dataset: torch.utils.data.Dataset = None

    """
    Fine tune the following parameters
    # Optimization parameters:
        *optimizer
        *learning rate
        *loss function
        *batch size
    # model parameters
        *number of layer
        *units per layer
        *dropout per layer
    """

    def __call__(self, trial: optuna.trial):
        trial_id = trial.number

        # Sample batchsize and create dataloaders
        batch_size = trial.suggest_int("batch_size", 4, 256)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # Sample model and loss function
        model = CustomMLP.define_model(trial)
        # model = MLP()
        model = model.cuda()

        loss_name = "MSELoss"
        loss_metric = getattr(torch.nn, loss_name)()

        # Sample training parameters
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        # Create trainer handler
        trainer_handler = TrainRegressionNet(
            train_loader,
            valid_loader,
            net=model,
            optimizer=optimizer,
            loss_metric=loss_metric,
            epochs=self.epochs,
            root=self.root / f"iter_{trial_id:04d}",
            gpu_boole=True,
            optimize_hyperparams=True,
            save=False,
        )

        # Train
        accuracy = trainer_handler.train_loop(trial, verbose=False)

        return accuracy


if __name__ == "__main__":
    # setup
    study_name = "regression_study1.pkl"
    root = Path(f"data/deep_learning_data/Studies/TestStudy2")
    data_dir = Path("data/03_replay_trajectory/d04-rec-10-traj01")
    train_batch_size = 64
    valid_batch_size = 64

    # ------------------------------------------------------------
    # Data loading and manipulation
    # ------------------------------------------------------------
    data_path = Path("data/deep_learning_data/random_dataset.txt")
    train_dataset = JointsDataset1(data_path, mode="train")
    normalizer = Normalizer(train_dataset.X)
    train_dataset.normalizer = normalizer
    valid_dataset = JointsDataset1(data_path, mode="valid", normalizer=normalizer)
    test_dataset = JointsDataset1(data_path, mode="test", normalizer=normalizer)

    # train_data = JointsRawDataset(data_dir, mode="train")
    # valid_data = JointsRawDataset(data_dir, mode="valid")
    # normalizer = Normalizer(*train_data[:])
    # train_dataset = JointsDataset(*train_data[:], normalizer)
    # valid_dataset = JointsDataset(*valid_data[:], normalizer)

    # ------------------------------------------------------------
    # Create hyperparameter study
    # ------------------------------------------------------------
    # Loss function
    loss_metric = torch.nn.L1Loss()

    regression_study = RegressionStudy1(
        root,
        study_name=study_name,
        epochs=220,
        loss_metric=None,
    )

    # Start study
    if (root / study_name).exists():
        print(f"Load: {root/study_name}")
        study = pickle.load(open(root / study_name, "rb"))
    else:
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner()
        )
    study.optimize(regression_study, n_trials=540, callbacks=[regression_study.save_state_cb])
