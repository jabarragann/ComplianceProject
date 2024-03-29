import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt

# Torch
import optuna
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# Custom
# from kincalib.Learning.Dataset import JointsDataset, JointsRawDataset, Normalizer
from kincalib.Learning.Dataset2 import JointsDataset1, Normalizer
from kincalib.Learning.Models import MLP, CustomMLP, JointCorrectionNet
from kincalib.Learning.Trainer import TrainRegressionNet
from torchsuite.HyperparameterTuner import OptuneStudyAbstract
from torchsuite.TrainingBoard import TrainingBoard

from kincalib.utils.CmnUtils import mean_std_str
from kincalib.utils.Logger import Logger


# def print_angle_dif(error_means, error_std):
#     log.info(f"Joint 1 mean difference (deg): {mean_std_str(error_means[0]*180/np.pi,error_std[0]*180/np.pi)}")
#     log.info(f"Joint 2 mean difference (deg): {mean_std_str(error_means[1]*180/np.pi,error_std[1]*180/np.pi)}")
#     log.info(f"Joint 3 mean difference (m):   {mean_std_str(error_means[2],error_std[2])}")
#     log.info(f"Joint 4 mean difference (deg): {mean_std_str(error_means[3]*180/np.pi,error_std[3]*180/np.pi)}")
#     log.info(f"Joint 5 mean difference (deg): {mean_std_str(error_means[4]*180/np.pi,error_std[4]*180/np.pi)}")
#     log.info(f"Joint 6 mean difference (deg): {mean_std_str(error_means[5]*180/np.pi,error_std[5]*180/np.pi)}")
#     log.info("")


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


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    epochs = 4500
    study_name = "regression_study1.pkl"
    study_root = Path(f"data/deep_learning_data/Studies/TestStudy2/")
    # root = study_root / "best_model5_temp" #epochs=260
    # root = study_root / "best_model7_psm2"  # epochs=4660
    root = study_root / "model13_rec20v2clean_6er"  # epochs=4660
    # root = study_root / "best_model09_6er"  # epochs=4660
    # root = study_root / "model10_rec23_6er"  # epochs=4660
    output_units = 6
    network_output = "error"
    assert output_units in [3, 6]
    # data_path = Path("data/deep_learning_data/random_dataset.txt")
    data_path = Path("data/deep_learning_data/rec20_data_v2_clean.csv")
    dataset_def_name = "dataset_def_" + data_path.with_suffix("").name + ".json"
    dataset_def_name = data_path.parent / dataset_def_name

    log = Logger("main_retrain").log

    if not root.exists():
        root.mkdir(parents=True)
    shutil.copy(dataset_def_name, root / dataset_def_name.name)

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reset", action="store_true", help="reset training")

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load hyperparameter study & best trial
    # ------------------------------------------------------------
    if (study_root / study_name).exists():
        log.info(f"Load: {study_root/study_name}")
        study = pickle.load(open(study_root / study_name, "rb"))

    best_trial: optuna.Trial = study.best_trial
    log.info(f"Total number of trials {len(study.trials)}")
    log.info(f"Best acc: {best_trial.value}")

    params_dict = {}
    for key, value in best_trial.params.items():
        log.info("{:20s}: {}".format(key, value))
        params_dict[key] = value
    params_dict["n_out"] = output_units
    params_dict["output"] = network_output

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
    # test_dataset = JointsDataset1(data_path, mode="test", normalizer=normalizer)
    log.info(f"Train dataset size: {len(train_dataset)}")
    log.info(f"Valid dataset size: {len(valid_dataset)}")

    # train_data = JointsRawDataset(data_dir, mode="train")
    # valid_data = JointsRawDataset(data_dir, mode="valid")
    # normalizer = Normalizer(*train_data[:])
    # train_dataset = JointsDataset(*train_data[:], normalizer)
    # valid_dataset = JointsDataset(*valid_data[:], normalizer)

    train_dataloader = DataLoader(
        train_dataset, batch_size=best_trial.params["batch_size"], shuffle=True
    )
    validation_dataloader = DataLoader(
        valid_dataset, batch_size=best_trial.params["batch_size"], shuffle=False
    )

    # ------------------------------------------------------------
    # Network & optimizer
    # ------------------------------------------------------------
    # model = CustomMLP.define_model(best_trial)
    model = JointCorrectionNet(model_def=params_dict)
    model.model_def_to_json(root)
    model = model.cuda()
    model = model.train()
    # Loss
    loss_name = "MSELoss"  # best_trial.suggest_categorical("loss", ["L1Loss", "MSELoss"])
    loss_metric = getattr(torch.nn, loss_name)()
    # Optimization parameters
    optimizer_name = best_trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = best_trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # ------------------------------------------------------------
    # Train network
    # ------------------------------------------------------------
    trainer_handler = TrainRegressionNet(
        train_dataloader,
        validation_dataloader,
        model,
        optimizer,
        loss_metric,
        epochs=epochs,
        root=root,
        gpu_boole=True,
        save=True,
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
    # Model Evaluation
    # ------------------------------------------------------------
    # TODO: Show results as table
    model.eval()
    log.info("calculating accuracy after training... (With model in eval mode)")
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
    valid_predicted_joints = model(valid_robot_state.cuda())

    valid_tracker_joints = valid_tracker_joints.cpu().data.numpy()
    valid_predicted_joints = valid_predicted_joints.cpu().data.numpy()
    valid_robot_state = normalizer.reverse(valid_robot_state)
    valid_robot_state = valid_robot_state.cpu().data.numpy()

    # ------------------------------------------------------------
    # Angle Differences
    # ------------------------------------------------------------
    print_angle_dif = print_angle_dif_full if full_output else print_angle_dif_partial

    loss_value = abs(valid_predicted_joints - valid_tracker_joints).mean(axis=0).mean()
    error = valid_tracker_joints - valid_predicted_joints
    error_means = error.mean(axis=0)
    error_std = error.std(axis=0)
    log.info(
        f"Angle difference between predicted joints and ground-truth (Tracker). Loss values: {loss_value:0.06f}"
    )
    print_angle_dif(error_means, error_std)

    robot_state_slice = slice(0, 6) if full_output else slice(3, 6)
    loss_value = (
        abs(valid_robot_state[:, robot_state_slice] - valid_tracker_joints).mean(axis=0).mean()
    )
    error = valid_tracker_joints - valid_robot_state[:, robot_state_slice]
    error_means = error.mean(axis=0)
    error_std = error.std(axis=0)
    log.info(
        f"Angle difference between robot measured joints and ground-truth (Tracker). Loss values: {loss_value:0.06f}"
    )
    print_angle_dif(error_means, error_std)

    # Training plots
    training_board = TrainingBoard(trainer_handler.checkpoint_handler, root=root)
    fig = training_board.training_plots()

    image_format = "svg"
    image_name = root / "training_plots.svg"
    fig.savefig(image_name, format=image_format, dpi=1200)

    plt.show()
