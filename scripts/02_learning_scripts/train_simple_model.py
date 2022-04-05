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

# Custom
from kincalib.utils.Logger import Logger
from kincalib.Learning.Dataset import Normalizer, JointsRawDataset, JointsDataset
from kincalib.Learning.Models import MLP
from kincalib.Learning.Trainer import TrainRegressionNet
from torchsuite.TrainingBoard import TrainingBoard

## Parameters
# Dataloaders
batch_size = 256
num_workers = 3
# Optimization
learning_rate = 0.001
shuffle = True
# Others
random_seed = 0
init_epoch = 0

log = Logger("train_segmentation").log
gpu_boole = torch.cuda.is_available()

if __name__ == "__main__":
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--round", type=int, help="Model training round", default=4)
    parser.add_argument("-c", "--loadcheckpoint", type=bool, default=True, help="Resume training from checkpoint")
    parser.add_argument("-e", "--epochs", type=int, default=552, help="epochs")
    args = parser.parse_args()
    epochs = args.epochs
    training_round = args.round
    # datapaths
    root = Path(f"data/ModelsCheckpoints/T{training_round:02d}/")
    if not root.exists():
        root.mkdir(parents=True)
    # Setup
    log = Logger("regression").log
    gpu_boole = torch.cuda.is_available()
    log.info(f"GPU available: {gpu_boole}")
    log.info(f"Model training round {training_round}")
    log.info(f"epochs {epochs}")

    # General Data Directory
    data_dir = Path("data/03_replay_trajectory/d04-rec-10-traj01")

    # ------------------------------------------------------------
    # Data loading and manipulation
    # ------------------------------------------------------------
    train_data = JointsRawDataset(data_dir, mode="train")
    valid_data = JointsRawDataset(data_dir, mode="valid")
    normalizer = Normalizer(*train_data[:])
    train_dataset = JointsDataset(*train_data[:], normalizer)
    valid_dataset = JointsDataset(*valid_data[:], normalizer)

    # Create Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    log.info(f"Training size   {len(train_dataset)}")
    log.info(f"Validation size {len(valid_dataset)}")

    # ------------------------------------------------------------
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
    # Train network
    # ------------------------------------------------------------
    # Create trainer handler
    trainer_handler = TrainRegressionNet(
        train_dataloader,
        validation_dataloader,
        net,
        optimizer,
        loss_metric,
        epochs,
        root=root,
        gpu_boole=True,
    )

    # Check for checkpoints
    if args.loadcheckpoint:
        checkpath = root / "final_checkpoint.pt"
        if checkpath.exists():
            trainer_handler.load_checkpoint(checkpath)
            log.info(f"resuming training from epoch {trainer_handler.init_epoch}")

    # Accuracy before training
    acc = trainer_handler.calculate_acc(train_dataloader)
    log.info(f"DSC before training {acc:0.04}")

    ## Train model
    loss_batch_store = trainer_handler.train_loop(verbose=False)
    log.info("*" * 30)

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------
    trainer_handler.save_training_parameters(root)

    # Save model
    torch.save(model.state_dict(), root / "model_state.pt")
    trainer_handler.save_checkpoint("final_checkpoint.pt")

    # ------------------------------------------------------------
    # Results after training
    # ------------------------------------------------------------
    log.info("calculating accuracy after training...")
    train_acc = trainer_handler.calculate_acc(trainer_handler.train_loader)
    train_loss = trainer_handler.calculate_loss(trainer_handler.train_loader)
    valid_acc = trainer_handler.calculate_acc(validation_dataloader)
    valid_loss = trainer_handler.calculate_loss(validation_dataloader)

    results = {
        "train_acc": train_acc,
        "train_loss": train_loss,
        "valid_acc": valid_acc,
        "valid_loss": valid_loss,
    }

    with open(root / f"results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Average training results")
    log.info(f"Average training acc:   {train_acc:0.06f}")
    log.info(f"Average training loss:  {train_loss:0.06f}")
    log.info(f"Average valid acc:      {valid_acc:0.06f}")
    log.info(f"Average valid loss:     {valid_loss:0.06f}")

    # ------------------------------------------------------------
    # Result plots
    # ------------------------------------------------------------
    training_board = TrainingBoard(trainer_handler.checkpoint_handler, root=root)
    training_board.training_plots()
