from pathlib import Path
from typing import Optional, TYPE_CHECKING # Import necessary types

from accelerate import PartialState # Import PartialState
from wandb.sdk.wandb_run import Run as WandbRun # Import WandbRun

from treetune.common import Registrable, FromParams

from treetune.logging_utils import get_logger

from treetune.runtime.base_runtime import Runtime

logger = get_logger(__name__)


class Trainer(Registrable, FromParams):
    """
    Base class for all trainers.
    Handles common initialization like setting up distributed state, paths, logging, and runtime.
    """
    def __init__(
            self,
            # Arguments expected based on the DeepSpeedPolicyTrainer super() call
            distributed_state: PartialState,
            experiment_root: Path,
            cloud_logger: Optional[WandbRun] = None,
            runtime: Optional["Runtime"] = None,
            # Add **kwargs to catch any unexpected arguments gracefully
            **kwargs
            ):
        """
        Initializes the base Trainer.

        Args:
            distributed_state: The state object for distributed training.
            experiment_root: The root directory for the experiment.
            cloud_logger: Optional logger (e.g., WandB run object).
            runtime: Optional reference to the runtime instance.
        """
        # --- Call super().__init__() for the parent classes (Registrable, FromParams) ---
        # --- Make sure this call takes NO arguments to avoid the error with object.__init__ ---
        super().__init__()

        # --- Store the passed arguments as instance attributes ---
        self.distributed_state = distributed_state
        self.experiment_root = experiment_root
        self.cloud_logger = cloud_logger
        self.runtime = runtime # This makes self.runtime available to subclasses

        # --- Common setup like checkpoints directory ---
        self.checkpoints_dir = self.experiment_root / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Initialized base Trainer. Runtime object: {self.runtime}")