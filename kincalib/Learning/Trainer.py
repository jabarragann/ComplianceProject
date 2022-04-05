from abc import ABC, abstractclassmethod
import json
from pathlib import Path
import numpy as np
import time
from rich.progress import track
from dataclasses import dataclass

# Torch
import optuna
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# Custom
from torchsuite.Trainer import Trainer
from kincalib.utils.Logger import Logger
from pytorchcheckpoint.checkpoint import CheckpointHandler

log = Logger(__name__).log
np.set_printoptions(precision=4, suppress=True, sign=" ")


# @dataclass
# class Trainer(ABC):
#     train_loader: DataLoader
#     valid_loader: DataLoader
#     net: nn.Module
#     optimizer: nn.Module
#     loss_metric: nn.Module
#     epochs: int
#     root: Path
#     gpu_boole: bool = True
#     optimize_hyperparams: bool = False
#     save: bool = True

#     def __post_init__(self):
#         self.init_epoch = 0
#         self.final_epoch = 0
#         self.batch_count = 0
#         self.best_valid_acc = 0.0
#         self.checkpoint_handler = CheckpointHandler()

#     def train_loop(self, trial: optuna.Trial = None, verbose=True):
#         log.info(f"Starting Training")
#         valid_acc = 0
#         for epoch in track(range(self.init_epoch, self.epochs), "Training network"):
#             time1 = time.time()
#             loss_sum = 0
#             total = 0

#             # Batch loop
#             for i, (x, y) in enumerate(self.train_loader):
#                 if self.gpu_boole:
#                     x = x.cuda()
#                     y = y.cuda()

#                 # loss calculation and gradient update:
#                 self.optimizer.zero_grad()
#                 outputs = self.net(x)
#                 loss = self.loss_metric(outputs, y)  # REMEMBER loss(OUTPUTS,LABELS)
#                 loss.backward()
#                 self.optimizer.step()  # Update parameters

#                 # End of batch stats
#                 self.checkpoint_handler.store_running_var(
#                     var_name="train_loss_batch", iteration=self.batch_count, value=loss.cpu().data.item()
#                 )
#                 self.batch_count += 1
#                 loss_sum += loss * y.shape[0]
#                 total += y.shape[0]

#             # End of epoch statistics
#             train_loss = loss_sum / total
#             train_loss = train_loss.cpu().item()
#             train_acc = self.calculate_acc(self.train_loader)
#             valid_acc = self.calculate_acc(self.valid_loader)
#             self.checkpoint_handler.store_running_var(var_name="train_loss", iteration=epoch, value=train_loss)
#             self.checkpoint_handler.store_running_var(var_name="train_acc", iteration=epoch, value=train_acc)
#             self.checkpoint_handler.store_running_var(var_name="valid_acc", iteration=epoch, value=valid_acc)
#             self.final_epoch = epoch
#             self.init_epoch = self.final_epoch

#             # Saving models
#             if valid_acc > self.best_valid_acc and self.save:
#                 log.info("saving best validation model")
#                 self.best_valid_acc = valid_acc
#                 self.save_checkpoint("best_checkpoint.pt")
#             if self.save:
#                 self.save_checkpoint("final_checkpoint.pt")

#             # Print epoch information
#             time2 = time.time()
#             if verbose:
#                 log.info(f"Epoch {epoch}/{self.epochs-1}:")
#                 log.info(f"Elapsed time for epoch: { time2 - time1:0.04f} s")
#                 log.info(f"Training loss:     {train_loss:0.8f}")
#                 log.info(f"Training accuracy: {train_acc:0.6f}")
#                 log.info(f"Valid accuracy:    {valid_acc:0.6f}")
#                 log.info(f"*" * 30)

#             # Optune callbacks
#             if self.optimize_hyperparams:
#                 # Optune prune mechanism
#                 trial.report(valid_acc, epoch)
#                 if trial.should_prune():
#                     raise optuna.exceptions.TrialPruned()

#         return valid_acc

#     @abstractclassmethod
#     def calculate_acc(self, dataloader: DataLoader):
#         pass

#     def calculate_loss(self, dataloader: DataLoader):
#         loss_sum = 0
#         total = 0
#         with torch.no_grad():
#             for x, y in dataloader:
#                 if self.gpu_boole:
#                     x = x.cuda()
#                     y = y.cuda()
#                 outputs = self.net(x)
#                 loss_sum += self.loss_metric(outputs, y) * y.shape[0]
#                 total += y.shape[0]

#             loss = loss_sum / total
#         return loss.cpu().data.item()

#     def save_training_parameters(self, root: Path):
#         train_params = {
#             "lr": self.optimizer.param_groups[0]["lr"],
#             "epochs": self.epochs,
#             "batch": self.train_loader.batch_size,
#             "opt": {"name": str(type(self.optimizer)), "parameters": self.optimizer.defaults},
#         }

#         with open(root / "trainer_parameters.json", "w") as f:
#             json.dump(train_params, f, indent=3)

#     def load_checkpoint(self, root: Path):
#         self.checkpoint_handler, self.net, self.optimizer = CheckpointHandler.load_checkpoint_with_model(
#             root, self.net, self.optimizer
#         )
#         self.init_epoch = self.checkpoint_handler.iteration + 1
#         self.batch_count = self.checkpoint_handler.batch_count
#         self.best_valid_acc = self.checkpoint_handler.get_var("best_valid_acc")

#     def save_checkpoint(self, filename):
#         # Save training params json
#         self.save_training_parameters(self.root)

#         # Save Checkpoint
#         self.checkpoint_handler.store_var(var_name="best_valid_acc", value=self.best_valid_acc)
#         self.checkpoint_handler.store_var(var_name="last_epoch", value=self.final_epoch)
#         self.checkpoint_handler.save_checkpoint(
#             checkpoint_path=self.root / filename,
#             iteration=self.final_epoch,
#             batch_count=self.batch_count,
#             model=self.net,
#             optimizer=self.optimizer,
#         )

#     def __str__(self):
#         train_params = {
#             "lr": self.optimizer.param_groups[0]["lr"],
#             "epochs": self.epochs,
#             "batch": self.batch_size,
#             "opt": {"name": str(type(self.optimizer)), "parameters": self.optimizer.defaults},
#         }
#         return json.dumps(train_params, indent=3)


@dataclass
class TrainRegressionNet(Trainer):
    def __post_init__(self):
        self.l1_loss = torch.nn.L1Loss()
        super().__post_init__()

    @torch.no_grad()
    def calculate_acc(self, dataloader: DataLoader):
        mean_loss = 0
        total = 0
        for x, mask in dataloader:
            if self.gpu_boole:
                x = x.cuda()
                mask = mask.cuda()
            output = self.net(x)

            mean_loss += self.l1_loss(output, mask) * mask.shape[0]
            total += mask.shape[0]

        total_loss = mean_loss / total
        return total_loss.cpu().data.item()


if __name__ == "__main__":
    pass
