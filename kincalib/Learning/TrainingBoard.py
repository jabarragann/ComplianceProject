from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Custom
from kincalib.Learning.Trainer import Trainer
from kincalib.utils.Logger import Logger
from pytorchcheckpoint.checkpoint import CheckpointHandler

log = Logger(__name__).log


class TrainingBoard:
    def __init__(self, checkpoint: CheckpointHandler = None, root=None) -> None:
        if checkpoint is None:
            raise Exception("TrainingBoard requires a valid checkpoint")
        # Todo: Raise exception if the checkpoint does not has the required attributes

        self.checkpoint = checkpoint
        # self.init_from_root(root)

    def init_from_root(self, root):
        self.train_acc_store = np.load(root / "train_acc.npy")
        self.valid_acc_store = np.load(root / "valid_acc.npy")
        self.loss_batch_store = np.load(root / "train_loss.npy")

    def create_acc_plot(self, ax):
        if ax is None:
            ax = TrainingBoard.create_ax()
        train_acc = self.checkpoint.get_running_var("train_acc")
        keys = [*train_acc]
        values = [*train_acc.values()]
        # ax.plot(self.train_acc_store, color="blue", label="train acc")
        ax.plot(keys, values, color="blue", label="train acc")

        valid_acc = self.checkpoint.get_running_var("valid_acc")
        keys = [*valid_acc]
        values = [*valid_acc.values()]
        ax.plot(keys, values, color="orange", label="valid acc")
        # ax.plot(self.valid_acc_store, color="orange", label="valid acc")

        ax.set_title("Accuracy curves")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid()
        return ax

    def create_loss_batch_plot(self, ax=None):
        if ax is None:
            ax = TrainingBoard.create_ax()
        ax.set_title("Batch training loss")
        train_loss = self.checkpoint.get_running_var("train_loss_batch")
        keys = [*train_loss]
        values = [*train_loss.values()]
        ax.plot(keys, values, label="train_loss", color="blue")
        # ax.plot(self.loss_batch_store, label="train_loss", color="orange")
        ax.set_xlabel("Minibatch Number")
        ax.set_ylabel("Sample-wise Loss")
        ax.legend()
        ax.grid()
        return ax

    def create_loss_epoch_plot(self, ax=None):
        if ax is None:
            ax = TrainingBoard.create_ax()
        ax.set_title("Epoch Training loss")
        train_loss = self.checkpoint.get_running_var("train_loss")
        keys = [*train_loss]
        values = [*train_loss.values()]
        ax.plot(keys, values, label="train_loss", color="blue")
        ax.set_xlabel("epoch")
        ax.set_ylabel("Sample-wise Loss")
        ax.legend()
        ax.grid()

        return ax

    def training_plots(self, plot=True):
        fig, axes = plt.subplots(1, 3)
        self.create_loss_batch_plot(ax=axes[0])
        self.create_loss_epoch_plot(ax=axes[1])
        self.create_acc_plot(ax=axes[2])
        if plot:
            plt.show()

    @staticmethod
    def create_ax():
        fig, ax = plt.subplots(1, 1)
        return ax
