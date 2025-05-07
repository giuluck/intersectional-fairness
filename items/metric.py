from abc import abstractmethod
from typing import Callable, List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, log_loss
from torch import nn, optim

from items.dataset import Dataset
from items.item import Item


class Metric:
    """Interface for a learning metric."""

    def __init__(self, name: str):
        """
        :param name:
            The name of the metric.
        """
        self._name: str = name

    @property
    def name(self) -> str:
        """The name of the metric."""
        return self._name

    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
        """Computes the metric between the input (x), output (y), and predictions (p)."""
        pass


class Loss(Metric):
    """Loss metric."""

    def __init__(self, classification: bool, name: Optional[str] = None):
        """
        :param classification:
            Whether the task is a regression or a classification one.

        :param name:
            The name of the metric.
        """
        metric, def_name = (log_loss, 'BCE') if classification else (mean_squared_error, 'MSE')
        super(Loss, self).__init__(name=def_name if name is None else name)
        self._metric: Callable = metric

    def __call__(self, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
        return float(self._metric(y, p))


class Fairness(Metric, Item):
    """A fairness metric for intersectional fairness."""

    eps: float = 1e-9
    """Tolerance used to avoid divisions by zero or log zero."""

    @classmethod
    def last_edit(cls) -> str:
        return "2025-04-24 00:00:00"

    @staticmethod
    def _network(dim: int, seed: int) -> Tuple[nn.Module, optim.Optimizer]:
        # build the neural network with given input dim and initialize it with fixed seed to guarantee reproducibility
        layers = []
        generator = torch.Generator(device='cpu').manual_seed(seed)
        for linear in [nn.Linear(dim, 16), nn.Linear(16, 16), nn.Linear(16, 8), nn.Linear(8, 1)]:
            nn.init.xavier_uniform_(linear.weight, generator=generator)
            nn.init.constant_(linear.bias, val=0.01)
            layers += [linear, nn.ReLU()]
        net = nn.Sequential(*layers[:-1])
        return net, optim.Adam(net.parameters(), lr=0.0005)

    @staticmethod
    def _gedi(a: torch.Tensor, b: torch.Tensor, net: nn.Module, opt: optim.Optimizer, epochs: int) -> torch.Tensor:
        """Implementation of a procedure to compute the GeDI indicator described in Equation 12 of the paper
        "Generalized Disparate Impact for Configurable Fairness Solutions in ML" by Luca Giuliani, Eleonora Misino,
        and Michele Lombardi (https://proceedings.mlr.press/v202/giuliani23a/giuliani23a.pdf) with a minor variation
        on the imposed constraint in order to use the estimation technique described in Algorithm 1 of the paper
        "Fairness-Aware Neural Renyi Minimization for Continuous Features" by Vincent Grari, Sylvain Lamprier,
        and Marcin Detyniecki (https://www.ijcai.org/proceedings/2020/0313.pdf) with ad-hoc adjustments."""
        b = torch.squeeze(b - b.mean())
        b_detach = b.detach()
        # train the adversarial networks (use detached vector to avoid nested gradient loops)
        for _ in range(epochs):
            opt.zero_grad()
            fa = net(a).squeeze()
            std_fa, mean_fa = torch.std_mean(fa, correction=0)
            loss = -torch.dot(fa - mean_fa, b_detach) / (std_fa + Fairness.eps) / len(fa)
            loss.backward()
            opt.step()
        # compute the indicator value as the (mean) vector product of standardized outputs
        fa = net(a).squeeze()
        std_fa, mean_fa = torch.std_mean(fa, correction=0)
        return torch.dot(fa - mean_fa, b) / (std_fa + Fairness.eps) / len(fa)

    def _int(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Intersectional GeDI (computed using a multivariate neural network on the whole protected inputs)."""
        net, opt = self._info['network']
        epochs = self._info['epochs']
        # compute gedi on the whole dataset
        gedi = self._gedi(a=a, b=b, net=net, opt=opt, epochs=epochs)
        # change the number of epochs for subsequent calls
        self._info['epochs'] = 50
        return gedi

    def _ind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Independent GeDI (computed using multiple univariate neural networks on each protected input)."""
        gedi = []
        epochs = self._info['epochs']
        for i in range(len(self._protected)):
            net, opt = self._info[f'network-{i}']
            # compute gedi on a single input feature
            gedi.append(self._gedi(a=a[:, [i]], b=b, net=net, opt=opt, epochs=epochs))
        # change the number of epochs for subsequent calls
        self._info['epochs'] = 50
        # return maximal value
        return torch.stack(gedi).max()

    # noinspection PyMethodMayBeStatic
    def _edf(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Implementation of the formula described in Lemma VIII.1 of the paper "An Intersectional Definition of
        Fairness" by James R. Foulds, Rashidul Islam, Kamrun Naher Keya, and Shimei Pan
        (https://arxiv.org/pdf/1807.08362)"""
        # categorize each unique row of the input tensor as a group
        group = a.unique(dim=0, return_inverse=True)[1]
        # keep a list of the empirical differential fairness (EDF) by browsing all the existing groups and computing
        # the log-probability (log-proportion) for each of them -- we add an eps to avoid log(0) = -inf
        edf = torch.stack([torch.log(b[group == g].mean() + self.eps) for g in range(group.max() + 1)])
        # since epsilon-DF is satisfied if the difference between the most and least privileged groups, we return
        # epsilon as this difference
        return edf.max() - edf.min()

    # noinspection PyMethodMayBeStatic
    def _spsf(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Implementation of the formula described in Definition 2.1 of the paper "Preventing Fairness Gerrymandering:
        Auditing and Learning for Subgroup Fairness" by Michael Kearns, Seth Neel, Aaron Roth, and Zhiwei Steven Wu
        (https://proceedings.mlr.press/v80/kearns18a.html)"""
        # categorize each unique row of the input tensor as a group
        group = a.unique(dim=0, return_inverse=True)[1]
        # compute the acceptance rate SP(D) on the whole dataset as the mean value (assuming 0/1 values only)
        spd = b.mean()
        # keep a list of the statistical parity subgroup fairness (SPSF) for each group
        spsf = []
        # browse all the existing groups
        for g in range(group.max() + 1):
            mask = group == g
            # compute the acceptance rates SP(D, g) on the group subset
            spdg = b[mask].mean()
            # compute the alpha factor, i.e., the proportional representation of the group
            alpha = mask.to(dtype=torch.float).mean()
            # compute the group SPSF
            spsf.append(alpha * torch.abs(spdg - spd))
        # since gamma-SP is satisfied if all the subgroups are gamma-fair, we return gamma as the maximum of the SPSFs
        return torch.stack(spsf).max()

    def __init__(self,
                 indicator: str,
                 protected: List[int],
                 scale: Optional[Dataset] = None,
                 name: Optional[str] = None):
        """
        :param indicator:
            The estimation indicator.

        :param protected:
            The indices of the protected features.

        :param scale:
            Whether to compute the percentage of unfairness with respect to a given dataset.

        :param name:
            The name of the metric.
        """
        if indicator == 'int':
            info = {'network': Fairness._network(dim=len(protected), seed=0), 'epochs': 1000}
            routine = self._int
        elif indicator == 'ind':
            info = {**{f'network-{i}': Fairness._network(dim=1, seed=i) for i in range(len(protected))}, 'epochs': 1000}
            routine = self._ind
        elif indicator == 'edf':
            info = dict()
            routine = self._edf
        elif indicator == 'spsf':
            info = dict()
            routine = self._spsf
        else:
            raise AssertionError(f"Unknown indicator '{indicator}'")

        # if no dataset is passed, scale = 1.0, otherwise it is computed on the bases of the input dataset
        if scale is None:
            scale = 1.0
        else:
            scale = Fairness(indicator=indicator, protected=protected).__call__(x=scale.x, y=scale.y, p=scale.y)

        super(Fairness, self).__init__(name=indicator.upper() if name is None else name)
        self._scale: float = scale
        self._protected: List[int] = protected.copy()
        self._routine: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = routine
        self._info: Dict[str, Any] = info

    @property
    def configuration(self) -> Dict[str, Any]:
        return {'indicator': self.indicator, 'protected': self._protected}

    @property
    def indicator(self) -> str:
        """The estimation indicator."""
        return self._name.lower()

    def compute(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Computes the fairness metric.

        :param a:
            The protected variables tensors.

        :param b:
            The target variable tensor.

        :return:
            The (un)fairness value obtained.
        """
        # select protected features only and then divide by scaling factor
        return self._routine(a[:, self._protected], b) / self._scale

    def __call__(self, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
        a = torch.tensor(x, dtype=torch.float)
        b = torch.tensor(p, dtype=torch.float)
        return self.compute(a=a, b=b).numpy(force=True).item()
