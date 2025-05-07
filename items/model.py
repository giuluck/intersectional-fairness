import time
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Dict, Any
from typing import Tuple, Iterable
from typing import Union, Optional, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import Logger
from torch import nn, Tensor
from torch.autograd import Variable
from torch.optim import Optimizer, Adam
from torch.utils import data
from tqdm import tqdm

from items.dataset import Dataset
from items.metric import Fairness


@dataclass(frozen=True)
class Output:
    """Dataclass representing the output of a learning algorithm."""

    train_inputs: np.ndarray = field()
    """The training data."""

    train_target: np.ndarray = field()
    """The training targets."""

    train_predictions: np.ndarray = field()
    """The training predictions."""

    val_inputs: np.ndarray = field()
    """The validation data."""

    val_target: np.ndarray = field()
    """The validation targets."""

    val_predictions: np.ndarray = field()
    """The validation predictions."""

    additional: Dict[str, Any] = field(default_factory=dict)
    """A dictionary of additional results."""


@dataclass(frozen=True)
class ConstraintInfo:
    """Dataclass representing the information of a constrained protected feature during training."""

    multiplier: Variable = field()
    """The multiplier for this feature."""

    f_net: Optional[nn.Module] = field()
    """The indicator net F used for warm starting."""

    g_net: Optional[nn.Module] = field()
    """The indicator net G used for warm starting."""


class Data(data.Dataset):
    """Default dataset for Torch."""

    def __init__(self, x: Iterable, y: Iterable):
        self.x: Tensor = torch.tensor(np.array(x), dtype=torch.float32)
        self.y: Tensor = torch.tensor(np.array(y), dtype=torch.float32).reshape((len(self.x), -1))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class History(Logger):
    def __init__(self):
        self._results: List[Dict[str, float]] = []

    @property
    def results(self) -> Dict[str, List[float]]:
        return {str(k): list(v) for k, v in pd.DataFrame(self._results).items()}

    @property
    def name(self) -> Optional[str]:
        return 'internal_logger'

    @property
    def version(self) -> Optional[Union[int, str]]:
        return 0

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if len(self._results) == step:
            self._results.append({'step': step})
        self._results[step].update(metrics)

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any):
        pass


class Storage(pl.Callback):
    """A callback that stores predictions at every step."""

    def __init__(self):
        self.results = {}

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: Dict[str, Any],
                           batch: Any,
                           batch_idx: int):
        # update the experiment with the external results
        step = trainer.global_step - 1
        train_inputs = trainer.train_dataloader.dataset.x.to(pl_module.device)
        val_inputs = trainer.val_dataloaders.dataset.x.to(pl_module.device)
        self.results[f'train_prediction_{step}'] = pl_module(train_inputs).numpy(force=True)
        self.results[f'val_prediction_{step}'] = pl_module(val_inputs).numpy(force=True)


class Progress(pl.Callback):
    """A callback that prints a progress bar throughout the learning process."""

    def __init__(self):
        self._pbar: Optional[tqdm] = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._pbar = tqdm(total=trainer.max_steps, desc='Model Training', unit='step')

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: Dict[str, Any],
                           batch: Any,
                           batch_idx: int):
        desc = 'Model Training (' + ' - '.join([f'{k}: {v:.4f}' for k, v in trainer.logged_metrics.items()]) + ')'
        self._pbar.set_description(desc=desc, refresh=True)
        self._pbar.update(n=1)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._pbar.close()
        self._pbar = None


class LagrangianDualModule(pl.LightningModule):
    """Implementation of the Lagrangian Dual Framework in Pytorch Lightning."""

    def __init__(self,
                 units: Iterable[int],
                 classification: bool,
                 threshold: float,
                 regularizer: Optional[Fairness]):
        """
        :param units:
            The neural network hidden units.

        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param threshold:
            The regularization threshold.

        :param regularizer:
            The indicator to be imposed as regularizer, or empty for unconstrained model.
        """
        super(LagrangianDualModule, self).__init__()

        # disable lightning manual optimization to deal with two optimizers
        self.automatic_optimization = False

        # build the layers by appending the final unit
        layers = []
        units = list(units)
        for inp, out in zip(units[:-1], units[1:]):
            layers.append(nn.Linear(inp, out))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(units[-1], 1))
        if classification:
            layers.append(nn.Sigmoid())

        self._model: nn.Sequential = nn.Sequential(*layers)
        self._loss: nn.Module = nn.BCELoss() if classification else nn.MSELoss()
        self._threshold: float = threshold
        self._regularizer: Optional[Fairness] = regularizer
        self._multiplier: Variable = Variable(torch.zeros(1), requires_grad=regularizer is not None)
        # if regularizer is None, create a dummy variable as multiplier with no gradient

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass on the model given the input (x)."""
        return self._model(x)

    # noinspection PyUnresolvedReferences
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """Performs a training step on the given batch."""
        # retrieve the data and the optimizers
        start = time.time()
        inp, out = batch
        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            task_opt, reg_opt = optimizers
            # patch to solve problem with lightning increasing the global step one time per optimizer
            reg_opt._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
            reg_opt._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        else:
            task_opt, reg_opt = optimizers, None
        # perform the standard loss minimization step
        task_opt.zero_grad()
        pred = self._model(inp)
        task_loss = self._loss(pred, out)
        # if there is a regularizer, compute it for the minimization step
        if self._regularizer is not None:
            reg = self._regularizer.compute(a=inp, b=pred)
            reg = torch.clip(reg - self._threshold, min=0.0)
        else:
            reg = 0.0
        # noinspection PyTypeChecker
        reg_loss = self._multiplier * reg
        # build the total minimization loss and perform the backward pass
        tot_loss = task_loss + reg_loss
        self.manual_backward(tot_loss)
        task_opt.step()
        # if there is a regularizer, run the maximization step (loss + regularization term with switched sign)
        if self._regularizer is not None:
            reg_opt.zero_grad()
            pred = self._model(inp)
            task_loss = self._loss(pred, out)
            reg = self._regularizer.compute(a=inp, b=pred)
            reg = torch.clip(reg - self._threshold, min=0.0)
            # noinspection PyTypeChecker
            reg_loss = self._multiplier * reg
            tot_loss = task_loss + reg_loss
            self.manual_backward(tot_loss)
            reg_opt.step()
        # return and log the information about the training
        self.log(name='time', value=time.time() - start, on_step=True, on_epoch=False, reduce_fx='sum')
        self.log(name='loss', value=tot_loss, on_step=True, on_epoch=False, reduce_fx='mean')
        self.log(name='task_loss', value=task_loss, on_step=True, on_epoch=False, reduce_fx='mean')
        self.log(name='reg_loss', value=reg_loss, on_step=True, on_epoch=False, reduce_fx='mean')
        self.log(name=f'reg', value=reg, on_step=True, on_epoch=False, reduce_fx='mean')
        # noinspection PyTypeChecker
        self.log(name=f'mul', value=self._multiplier, on_step=True, on_epoch=False, reduce_fx='mean')
        return tot_loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        inp, out = batch
        pred = self._model(inp)
        loss = self._loss(pred, out)
        self.log(name='val_loss', value=loss, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self) -> Union[Optimizer, Tuple[Optimizer, Optimizer]]:
        """Configures the optimizer for the MLP depending on whether there is a variable multiplier or not."""
        optimizer = Adam(params=self._model.parameters(), lr=1e-3)
        if isinstance(self._multiplier, Variable):
            # noinspection PyTypeChecker
            return optimizer, Adam(params=[self._multiplier], maximize=True, lr=1e-3)
        else:
            return optimizer


class Model:
    """A learning algorithm for constrained deep learning based on the Lagrangian Dual framework."""

    units: List[int] = [32, 32]
    """The number of units in the hidden layers."""

    epochs: int = 500
    """The number of training epochs."""

    threshold: float = 0.2
    """The relative threshold for the regularizers."""

    def __init__(self, indicator: Optional[str] = None):
        """
        :param indicator:
            The indicator to be imposed as regularizer (empty for unconstrained model).
        """
        self._indicator: Optional[str] = indicator

    def run(self, dataset: Dataset, fold: int, folds: int, seed: int) -> Output:
        # retrieve train and validation data from splits and set parameters
        trn, val = dataset.folds(k=folds, seed=seed)[fold]
        trn_data = Data(x=trn[dataset.features], y=trn[[dataset.target]])
        val_data = Data(x=val[dataset.features], y=val[[dataset.target]])
        # build model
        model = LagrangianDualModule(
            units=[len(dataset.features), *self.units],
            classification=dataset.classification,
            threshold=self.threshold,
            regularizer=None if self._indicator is None else Fairness(
                indicator=self._indicator,
                protected=dataset.protected_indices,
                scale=dataset
            )
        )
        # build trainer and callback
        history = History()
        progress = Progress()
        storage = Storage()
        trainer = pl.Trainer(
            accelerator='cpu',
            deterministic=True,
            min_steps=self.epochs,
            max_steps=self.epochs,
            logger=[history],
            callbacks=[progress, storage],
            num_sanity_val_steps=0,
            val_check_interval=1,
            log_every_n_steps=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False
        )
        # run fitting
        trainer.fit(
            model=model,
            train_dataloaders=data.DataLoader(trn_data, batch_size=len(trn), shuffle=True),
            val_dataloaders=data.DataLoader(val_data, batch_size=len(val), shuffle=False)
        )
        # store external files and return result
        return Output(
            train_inputs=trn_data.x.numpy(force=True),
            train_target=trn_data.y.numpy(force=True),
            train_predictions=model(trn_data.x).numpy(force=True),
            val_inputs=val_data.x.numpy(force=True),
            val_target=val_data.y.numpy(force=True),
            val_predictions=model(val_data.x).numpy(force=True),
            additional=dict(history=history.results, **storage.results)
        )
