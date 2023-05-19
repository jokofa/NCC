#
import logging
from typing import Union, List, Dict
from omegaconf import DictConfig

from lib.utils import CCPInstance
from lib.ltr.utils import collate_batch
from lib.ltr.task import BaseTask
from lib.ltr.vrp.vrp_model import VRPModel

logger = logging.getLogger('lightning')


class Task(BaseTask):
    """Outer PyTorch lightning module wrapper."""
    def __init__(self, cfg: Union[DictConfig, Dict]):
        super(Task, self).__init__(cfg)

    def _build_model(self):
        logger.info("building model...")
        self.model = VRPModel(
            input_dim=self.cfg.input_dim,
            embedding_dim=self.cfg.embedding_dim,
            decoder_type=self.cfg.decoder_type,
            node_encoder_args=self.cfg.node_encoder_args,
            center_encoder_args=self.cfg.center_encoder_args,
            decoder_args=self.cfg.decoder_args,
        )

    def training_step(self,
                      batch: List[CCPInstance],
                      batch_idx: int, *args, **kwargs):
        x, y = collate_batch(batch, device=self.device, dtype=self.dtype, vrp=True)
        y_hat, _ = self.model(
            nodes=x.nodes,
            centroids=x.centroids,
            medoids=x.medoids,
            c_mask=x.c_mask,
            edges=x.edges,
            weights=x.weights,
        )
        msk = ~x.c_mask     # mask dummy centers
        loss = self.loss_fn(y_hat[msk], y[msk])

        self.log(f"train_loss", loss,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 batch_size=self.cfg.train_batch_size)
        return loss

    def validation_step(self,
                        batch: List[CCPInstance],
                        batch_idx: int, *args, **kwargs):
        x, y = collate_batch(batch, device=self.device, dtype=self.dtype, vrp=True)
        y_hat, _ = self.model(
            nodes=x.nodes,
            centroids=x.centroids,
            medoids=x.medoids,
            c_mask=x.c_mask,
            edges=x.edges,
            weights=x.weights,
        )
        msk = ~x.c_mask  # mask dummy centers
        y_hat, y = y_hat[msk], y[msk]
        loss = self.loss_fn(y_hat, y)
        cur_acc = self.acc(y_hat, y)
        self.log(f"val_loss", loss,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 batch_size=self.cfg.val_batch_size)
        self.log(f"val_acc", self.acc,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 batch_size=self.cfg.val_batch_size)
        return cur_acc
