from __future__ import annotations

from pathlib import Path
from typing import Any, List
from kincalib.Entities.Msgs import MyJointState, MyPoseStamped

# Custom
from kincalib.Recording.DataRecord import CartesianRecord, JointRecord
from kincalib.utils.Logger import Logger

log = Logger(__name__).log


class RecordCollectionTemplate:
    def __init__(self, jp_filename: Path, cp_filename: Path) -> None:
        self.joint_record = JointRecord(jp_filename)
        self.pose_record = CartesianRecord(cp_filename)

    def add_new_robot_data(self, step, measured_js: MyJointState, setpoint_js: MyJointState, robot_cp: MyPoseStamped):
        self.joint_record.create_new_entry(step, measured_js, setpoint_js)
        self.pose_record.create_new_entry(step, "robot", "-1", robot_cp, setpoint_js)

    def add_fiducials_poses(self, step, setpoint_js: MyJointState, fiducials_cp: List[MyPoseStamped]):
        for fid_cp in fiducials_cp:
            self.pose_record.create_new_entry(step, "fiducial", "-1", fid_cp, setpoint_js)

    def add_tools_poses(
        self,
        step,
        setpoint_js: MyJointState,
        tools_cp: List[MyPoseStamped],
        tools_id_list: List[str],
    ):
        for tool_cp, id in zip(tools_cp, tools_id_list):
            self.pose_record.create_new_entry(step, "tool", id, tool_cp, setpoint_js)

    def save_data(self, safe_save=False):
        self.joint_record.to_csv(safe_save=safe_save)
        self.pose_record.to_csv(safe_save=safe_save)


class ExperimentRecordCollection(RecordCollectionTemplate):
    def __init__(
        self,
        root_dir: Path,
        mode: str = "calib",
        test_id: int = None,
        description: str = "",
    ) -> None:
        """_summary_

        Parameters
        ----------
        root_dir : _type_
            _description_
        mode : str, optional
            Operation mode, by default "calib". Needs to be either 'calib' or 'test'
        test_id : int, optional
            _description_, by default None
        description: str, optional
            one line experiment description
        """

        assert mode in ["calib", "test"], "mode needs to be calib or test"
        self.description = description
        self.root_dir = root_dir
        self.mode = mode
        self.test_id = test_id

        self.create_paths()
        self.cp_filename, self.jp_filename = self.obtain_filenames_for_records()

        super().__init__(self.jp_filename, self.cp_filename)

    def obtain_filenames_for_records(self):
        if self.mode == "calib":
            cp_filename = self.robot_files / ("robot_cp.csv")
            jp_filename = self.robot_files / ("robot_jp.csv")
        elif self.mode == "test":
            cp_filename = self.test_files / ("robot_cp.csv")
            jp_filename = self.test_files / ("robot_jp.csv")

        if cp_filename.exists() or jp_filename.exists():
            n = input(f"Data was found in directory {cp_filename.parent}. Press (y/Y) to overwrite. ")
            if not (n == "y" or n == "Y"):
                log.info("exiting the script")
                exit(0)
        return cp_filename, jp_filename

    def create_paths(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if self.mode == "calib":
            self.robot_files = self.root_dir / "robot_mov"
            self.robot_files.mkdir(parents=True, exist_ok=True)

        if self.mode == "test":
            assert self.test_id is not None, "undefined test id"
            self.test_files = self.root_dir / f"test_trajectories/{self.test_id:02d}"
            self.test_files.mkdir(parents=True, exist_ok=True)

        description_path = self.robot_files if self.mode == "calib" else self.test_files
        with open(description_path / "description.txt", "w") as f:
            f.write(self.description)
