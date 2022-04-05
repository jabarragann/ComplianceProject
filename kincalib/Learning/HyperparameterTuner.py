from pathlib import Path
import optuna
from abc import ABC, abstractclassmethod
import pickle
from dataclasses import dataclass

# Save state
class SaveStateCallback:
    def __init__(self, study_name: str, root: Path):
        self.root = root
        self.study_name = study_name

        if not self.root.exists():
            root.mkdir(parents=True)

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        pickle.dump(study, open(self.root / f"{self.study_name}", "wb"))


@dataclass
class OptuneStudyAbstract(ABC):
    root: Path
    study_name: str

    def __post_init__(self):
        self.save_state_cb = SaveStateCallback(self.study_name, self.root)

    @abstractclassmethod
    def __call__(self, trial):
        pass


if __name__ == "__main__":

    class OptuneStudy(OptuneStudyAbstract):
        def __call__(self, trial):
            return 0

    study_info = OptuneStudy(Path("./"), None, None, "test")
