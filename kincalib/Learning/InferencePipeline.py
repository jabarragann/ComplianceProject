import json
from pathlib import Path
import numpy as np

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


if __name__ == "__main__":

    root = Path(f"data/deep_learning_data/Studies/TestStudy2/")
    root = root / "best_model5_temp"

    inference_pipe = InferencePipeline(root)
