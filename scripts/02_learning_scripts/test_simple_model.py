# Python
import argparse
import json
from pathlib import Path
import torchvision.datasets as dataset
import matplotlib.pyplot as plt
import numpy as np

# Torch
import torchvision.transforms as transforms
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
from kincalib.utils.CmnUtils import mean_std_str

# Custom
from kincalib.utils.Logger import Logger
from kincalib.Learning.Dataset import Normalizer, JointsRawDataset, JointsDataset
from kincalib.Learning.Models import MLP
from kincalib.Learning.Trainer import TrainRegressionNet, Trainer
from kincalib.Learning.TrainingBoard import TrainingBoard
from pytorchcheckpoint.checkpoint import CheckpointHandler

## Parameters
# Dataloaders
train_batch_size = 5
valid_batch_size = 5
test_batch_size = 5
num_workers = 3
# Optimization
learning_rate = 0.001
shuffle = True
# Others
random_seed = 0

visualize = False
log = Logger("fashion").log
gpu_boole = torch.cuda.is_available()


def print_angle_dif(error_means, error_std):
    log.info(f"Joint 1 mean difference (deg): {mean_std_str(error_means[0]*180/np.pi,error_std[0]*180/np.pi)}")
    log.info(f"Joint 2 mean difference (deg): {mean_std_str(error_means[1]*180/np.pi,error_std[1]*180/np.pi)}")
    log.info(f"Joint 3 mean difference (m):   {mean_std_str(error_means[2],error_std[2])}")
    log.info(f"Joint 4 mean difference (deg): {mean_std_str(error_means[3]*180/np.pi,error_std[3]*180/np.pi)}")
    log.info(f"Joint 5 mean difference (deg): {mean_std_str(error_means[4]*180/np.pi,error_std[4]*180/np.pi)}")
    log.info(f"Joint 6 mean difference (deg): {mean_std_str(error_means[5]*180/np.pi,error_std[5]*180/np.pi)}")
    log.info("")


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--round", type=int, help="Model training round", default=1)
    args = parser.parse_args()
    training_round = args.round

    # datapaths
    root = Path(f"data/ModelsCheckpoints/T{training_round:02d}/")
    # root = Path(f"data/ModelsCheckpoints/Studies/TestStudy/best_model")

    # Setup
    log = Logger("regression").log
    gpu_boole = torch.cuda.is_available()

    # General Data Directory
    data_dir = Path("data/03_replay_trajectory/d04-rec-10-traj01")

    # ------------------------------------------------------------
    # Data loading and manipulation
    # ------------------------------------------------------------
    train_data_raw = JointsRawDataset(data_dir, mode="train")
    valid_data_raw = JointsRawDataset(data_dir, mode="valid")
    normalizer = Normalizer(*train_data_raw[:])
    train_dataset = JointsDataset(*train_data_raw[:], normalizer)
    valid_dataset = JointsDataset(*valid_data_raw[:], normalizer)

    # Create Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    validation_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)

    # Network & optimizer
    # ------------------------------------------------------------

    # Create Network
    model = MLP()
    # Initialize Optimizer and Learning Rate Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if gpu_boole:
        net = model.cuda()
    # Loss function
    loss_metric = torch.nn.L1Loss()

    # ------------------------------------------------------------
    # Load Checkpoint
    # ------------------------------------------------------------

    # Calculate final validation acc
    trainer_handler = TrainRegressionNet(
        train_dataloader,
        validation_dataloader,
        net,
        optimizer,
        loss_metric,
        epochs=-1,
        batch_size=train_batch_size,
        root=root,
        gpu_boole=True,
    )
    trainer_handler.load_checkpoint(root / "best_checkpoint.pt")

    # Printing settings
    log.info(f"GPU available: {gpu_boole}")
    log.info(f"Loading from training round {training_round}")
    log.info(f"Model trained for {trainer_handler.checkpoint_handler.iteration} epochs ")
    log.info(f"Training size   {len(train_dataset)}")
    log.info(f"Validation size {len(valid_dataset)}\n")

    # ------------------------------------------------------------
    # Model Evaluation
    # ------------------------------------------------------------

    log.info("calculating accuracy after training...")
    train_acc = trainer_handler.calculate_acc(trainer_handler.train_loader)
    train_loss = trainer_handler.calculate_loss(trainer_handler.train_loader)
    valid_acc = trainer_handler.calculate_acc(validation_dataloader)
    valid_loss = trainer_handler.calculate_loss(validation_dataloader)
    log.info(f"Average training results")
    log.info(f"Average training acc:   {train_acc:0.06f}")
    log.info(f"Average training loss:  {train_loss:0.06f}")
    log.info(f"Average valid acc:      {valid_acc:0.06f}")
    log.info(f"Average valid loss:     {valid_loss:0.06f}\n")

    # Calculate angle difference
    valid_robot_state, valid_tracker_joints = valid_dataset[:]
    valid_predicted_joints = net(torch.from_numpy(valid_robot_state).cuda())

    valid_predicted_joints = valid_predicted_joints.cpu().data.numpy()
    valid_robot_state = normalizer.reverse(valid_robot_state)

    # ------------------------------------------------------------
    # Angle Differences
    # ------------------------------------------------------------
    loss_value = abs(valid_predicted_joints - valid_tracker_joints).mean(axis=0).mean()
    error = valid_tracker_joints - valid_predicted_joints
    error_means = error.mean(axis=0)
    error_std = error.std(axis=0)
    log.info(f"Angle difference between predicted and ground-truth. Loss values: {loss_value:0.06f}")
    print_angle_dif(error_means, error_std)

    loss_value = abs(valid_robot_state[:, :6] - valid_tracker_joints).mean(axis=0).mean()
    error = valid_tracker_joints - valid_robot_state[:, :6]
    error_means = error.mean(axis=0)
    error_std = error.std(axis=0)
    log.info(f"Angle difference between input and ground-truth. Loss values: {loss_value:0.06f}")
    print_angle_dif(error_means, error_std)

    # Training plots
    training_board = TrainingBoard(trainer_handler.checkpoint_handler, root=root)
    training_board.training_plots()
    plt.show()
