import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from kincalib.Learning.Dataset2 import Normalizer
from kincalib.Learning.Models import JointCorrectionNet
from kincalib.utils.CmnUtils import mean_std_str
from kincalib.utils.Logger import Logger
import rospy

log = Logger("autonomy_utils").log


class InferencePipeline:
    def __init__(self, model_path: Path, device: str = "cuda"):

        self.device = torch.device(device)

        # Load model def
        with open(model_path / "model_def.json") as nf:
            self.model_def = json.load(nf)
        self.model = JointCorrectionNet(self.model_def)

        # Load normalizer
        with open(model_path / "normalizer.json") as nf:
            self.normalizer = Normalizer(xdata=None, state_dict=json.load(nf))

        # Load weigths
        checkpoint = torch.load(model_path / "final_checkpoint.pt", map_location="cpu")
        self.model.load_state_dict(checkpoint.model_state_dict)
        self.model.eval()

        if self.device.type == "cuda":
            log.info("Running model on GPU")
            self.model = self.model.cuda()

    def predict(self, x_data: np.ndarray):
        x_norm = self.normalizer(x_data)
        x_norm = torch.from_numpy(x_norm).cuda().float()
        x_norm = x_norm.cuda()
        pred = self.model(x_norm).cpu().detach().numpy()

        return pred

    def correct_joints(self, robot_state_df: pd.DataFrame) -> pd.DataFrame:
        """Correct joints using neural network model

        Parameters
        ----------
        robot_state_df : pd.DataFrame
            Dataframe containing the robot state. Expects the columns:

            ["step", "q1", "q2", "q3", "q4", "q5", "q6","t1", "t2", "t3", "t4", "t5", "t6"]

        Returns
        -------
        corrected_joints: pd.DataFrame
            Dataframe of corrected joints with columns:

            ["step","q1", "q2", "q3", "q4", "q5", "q6"]

        """

        cols = ["q1", "q2", "q3", "q4", "q5", "q6"] + ["t1", "t2", "t3", "t4", "t5", "t6"]

        valstopred = robot_state_df[cols].to_numpy().astype(np.float32)
        pred = self.predict(valstopred)

        # Recombine
        pred = np.hstack((robot_state_df["step"].to_numpy().reshape(-1, 1), pred))

        if pred.shape[1] == 3 + 1:  # Partial output
            corrected_joints_df = pd.DataFrame(pred, columns=["step", "q4", "q5", "q6"])
            # Add first three joints from robot
            corrected_joints_df = pd.merge(
                robot_state_df.loc[:, ["step", "q1", "q2", "q3"]], corrected_joints_df, on="step"
            )
        else:
            corrected_joints_df = pd.DataFrame(
                pred, columns=["step", "q1", "q2", "q3", "q4", "q5", "q6"]
            )
            # Replace network insertion joint by robot
            corrected_joints_df = pd.concat(
                (
                    corrected_joints_df[["step", "q1", "q2"]],
                    robot_state_df["q3"],
                    corrected_joints_df[["q4", "q5", "q6"]],
                ),
                axis=1,
            )

        return corrected_joints_df


if __name__ == "__main__":

    root = Path(f"data/deep_learning_data/Studies/TestStudy2/")
    root = root / "best_model5_temp"

    inference_pipe = InferencePipeline(root)
