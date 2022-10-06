# Python
import argparse
import json
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Torch
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from kincalib.Learning.Dataset2 import JointsDataset1, Normalizer
from kincalib.utils.CmnUtils import mean_std_str

# Custom
from kincalib.utils.Logger import Logger

# from kincalib.Learning.Dataset import Normalizer, JointsRawDataset, JointsDataset
from kincalib.Learning.Models import JointCorrectionNet
from kincalib.Learning.Trainer import TrainRegressionNet
from torchsuite.TrainingBoard import TrainingBoard

## Parameters
# Optimization
shuffle = True
random_seed = 0
init_epoch = 0

log = Logger("train_segmentation").log
gpu_boole = torch.cuda.is_available()


def print_angle_dif_partial(error_means, error_std):
    log.info(
        f"Joint 4 mean difference (deg): {mean_std_str(error_means[0]*180/np.pi,error_std[0]*180/np.pi)}"
    )
    log.info(
        f"Joint 5 mean difference (deg): {mean_std_str(error_means[1]*180/np.pi,error_std[1]*180/np.pi)}"
    )
    log.info(
        f"Joint 6 mean difference (deg): {mean_std_str(error_means[2]*180/np.pi,error_std[2]*180/np.pi)}"
    )
    log.info("")


def print_angle_dif_full(error_means, error_std):
    log.info(
        f"Joint 1 mean difference (deg): {mean_std_str(error_means[0]*180/np.pi,error_std[0]*180/np.pi)}"
    )
    log.info(
        f"Joint 2 mean difference (deg): {mean_std_str(error_means[1]*180/np.pi,error_std[1]*180/np.pi)}"
    )
    log.info(
        f"Joint 3 mean difference (mm):  {mean_std_str(error_means[2]*1000,error_std[2]*1000)}"
    )
    log.info(
        f"Joint 4 mean difference (deg): {mean_std_str(error_means[3]*180/np.pi,error_std[3]*180/np.pi)}"
    )
    log.info(
        f"Joint 5 mean difference (deg): {mean_std_str(error_means[4]*180/np.pi,error_std[4]*180/np.pi)}"
    )
    log.info(
        f"Joint 6 mean difference (deg): {mean_std_str(error_means[5]*180/np.pi,error_std[5]*180/np.pi)}"
    )
    log.info("")


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reset", action="store_true", help="reset training")
    parser.add_argument("-e", "--epochs", type=int, default=4500, help="epochs")
    args = parser.parse_args()
    epochs = args.epochs

    log = Logger("regression").log
    gpu_boole = torch.cuda.is_available()
    log.info(f"GPU available: {gpu_boole}")
    log.info(f"epochs {epochs}")

    # General Data Directory
    root = Path("icra2023-data/neuralnet/model")
    data_path = Path("icra2023-data/neuralnet/dataset/final_dataset_clean.csv")
    model_path = Path("icra2023-data/neuralnet")

    # ------------------------------------------------------------
    # Load hyperparameters
    # ------------------------------------------------------------
    with open(model_path / "model_def.json") as nf:
        model_def = json.load(nf)

    output_units = model_def["n_out"]
    network_output = model_def["output"]
    learning_rate = model_def["lr"]
    batch_size = model_def["batch_size"]

    # ------------------------------------------------------------
    # Data loading and manipulation
    # ------------------------------------------------------------
    full_output = output_units == 6
    train_dataset = JointsDataset1(
        data_path, mode="train", full_output=full_output, output=network_output
    )
    normalizer = Normalizer(train_dataset.X)
    normalizer.to_json(root / "normalizer.json")
    train_dataset.normalizer = normalizer
    pickle.dump(normalizer, open(root / "normalizer.pkl", "wb"))
    valid_dataset = JointsDataset1(
        data_path,
        mode="valid",
        normalizer=normalizer,
        full_output=full_output,
        output=network_output,
    )

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    log.info(f"Training size   {len(train_dataset)}")
    log.info(f"Validation size {len(valid_dataset)}")

    # ------------------------------------------------------------
    # Network & optimizer
    # ------------------------------------------------------------

    model = JointCorrectionNet(model_def)
    model.model_def_to_json(root)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if gpu_boole:
        net = model.cuda()
    loss_metric = torch.nn.MSELoss()

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
    if not args.reset:
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

    # ------------------------------------------------------------
    # Results after training
    # ------------------------------------------------------------
    model.eval()
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
    print("\n")

    # ------------------------------------------------------------
    # Angle Differences
    # ------------------------------------------------------------
    valid_robot_state, tracker_robot_dif = valid_dataset[:]
    valid_predicted_joints = model(valid_robot_state.cuda())

    tracker_robot_dif = tracker_robot_dif.cpu().data.numpy()
    valid_predicted_joints = valid_predicted_joints.cpu().data.numpy()
    valid_robot_state = normalizer.reverse(valid_robot_state)
    valid_robot_state = valid_robot_state.cpu().data.numpy()

    print_angle_dif = print_angle_dif_full if full_output else print_angle_dif_partial

    # Error between robot and ground truth
    robot_state_slice = slice(0, 6) if full_output else slice(3, 6)
    loss_value = abs(tracker_robot_dif).mean(axis=0).mean()
    error = tracker_robot_dif
    error_means = error.mean(axis=0)
    error_std = error.std(axis=0)
    log.info(
        f"Error between robot measured joints and ground-truth (Tracker). Loss values: {loss_value:0.06f}"
    )
    print_angle_dif(error_means, error_std)

    # Difference between robot corrected joints and tracker
    loss_value = abs(valid_predicted_joints - tracker_robot_dif).mean(axis=0).mean()
    error = tracker_robot_dif - valid_predicted_joints
    error_means = error.mean(axis=0)
    error_std = error.std(axis=0)
    log.info(
        f"Error between robot corrected joints and ground-truth (Tracker). Loss values: {loss_value:0.06f}"
    )
    print_angle_dif(error_means, error_std)

    # ------------------------------------------------------------
    # Result plots
    # ------------------------------------------------------------
    training_board = TrainingBoard(trainer_handler.checkpoint_handler, root=root)
    fig = training_board.training_plots()

    image_format = "svg"
    image_name = root / "training_plots.svg"
    fig.savefig(image_name, format=image_format, dpi=1200)

    plt.show()
