#
import logging
from typing import Union, Any, List, Optional, Dict
from omegaconf import DictConfig

import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm
from torch.utils.data import DataLoader
from pytorch_lightning.core.module import LightningModule

from lib.utils import get_lambda_decay, CCPInstance
from lib.ltr.utils import NPZProblemDataset

logger = logging.getLogger('lightning')


class BaseTask(LightningModule):
    """Outer PyTorch lightning module wrapper."""
    def __init__(self, cfg: Union[DictConfig, Dict]):
        super(BaseTask, self).__init__()

        self.cfg = cfg
        self.save_hyperparameters()

        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        #
        self.collate_fn = None
        self.loss_fn = None
        self.acc = None

    def forward(self, x):
        raise RuntimeError("Cannot use PLModule forward directly!")

    def setup(self, stage: Optional[str] = None):
        """Initial setup hook."""

        if stage is not None and stage.lower() == 'test':
            raise RuntimeError()
        # TRAINING setup ('fit')
        else:
            # load validation data
            logger.info("loading validation data...")
            self.val_dataset = NPZProblemDataset(
                npz_file_pth=self.cfg.val_dataset,
                knn=self.cfg.get("knn", 16),
            )
            # load training data
            logger.info("loading training data...")
            self.train_dataset = NPZProblemDataset(
                npz_file_pth=self.cfg.train_dataset,
                knn=self.cfg.get("knn", 16),
            )
            # set some attributes
            self.collate_fn = lambda x: x   # identity -> returning simple list of instances
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.acc = tm.Accuracy("binary")
            self._build_model()

    def _build_model(self):
        raise NotImplementedError()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # also need to load the original config
        cfg_ = checkpoint["hyper_parameters"]["cfg"]
        self.cfg.update(cfg_)
        self._build_model()

    def configure_optimizers(self):
        """Create optimizers and lr-schedulers for model."""
        # create optimizer
        opt = getattr(optim, self.cfg.optimizer)
        # provide model parameters and optionally trainable loss parameters to optimizer
        optimizer = opt(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            **self.cfg.optimizer_cfg
        )
        # create lr scheduler
        if self.cfg.scheduler_cfg.schedule_type is not None:
            decay_ = get_lambda_decay(**self.cfg.scheduler_cfg)
            lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, decay_)
            return (
               {'optimizer': optimizer, 'lr_scheduler': lr_scheduler},
            )
        return optimizer

    def train_dataloader(self) -> DataLoader:
        """Create the training data loader."""
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train_batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation data loader."""
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.val_batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def training_step(self,
                      batch: List[CCPInstance],
                      batch_idx: int, *args, **kwargs):
        raise NotImplementedError()

    def validation_step(self,
                        batch: List[CCPInstance],
                        batch_idx: int, *args, **kwargs):
        raise NotImplementedError()
