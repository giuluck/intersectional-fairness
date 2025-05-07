from abc import abstractmethod
from typing import List, Set, Optional, Tuple
from typing import final, Dict, Any

import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from items.item import Item


class DatasetInfo:
    """Class containing dataset information."""

    classification: bool
    """Whether this is a classification or a regression task."""

    target: str
    """The name of the target feature."""

    numeric: Set[str]
    """The set of numeric features."""

    binary: Set[str]
    """The set of binary features."""

    def __init__(self):
        self._data: Optional[pd.DataFrame] = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            # load data, convert all numeric columns to float, and drop columns/rows with nan values
            data = self.load().astype({c: float for c in self.numeric})
            data = data.dropna(axis=1, thresh=int(0.9 * len(data))).dropna(axis=0)
            # standardize numeric features (or normalize if target) and categorize binary variables
            for column, values in data.items():
                if column in self.numeric:
                    if column == self.target:
                        data[column] = (values - values.min()) / (values.max() - values.min())
                    else:
                        data[column] = (values - values.mean()) / values.std(ddof=0)
                elif column in self.binary:
                    data[column] = values.astype('category').cat.codes.astype(float)
            # the remaining (categorical) features are one-hot encoded
            self._data = data.pipe(pd.get_dummies).astype(float).reset_index(drop=True)
        return self._data

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Loads the original dataframe.

        :return:
            The original dataframe of the benchmark.
        """
        pass

    @abstractmethod
    def protected(self, continuous: bool) -> List[str]:
        """The names of the protected features.

        :param continuous:
            Whether the protected attributes can include continuous features as well.

        :return:
            A list containing the names of the protected features.
        """
        pass


class Adult(DatasetInfo):
    classification = False
    target = 'income'
    numeric = {'age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week'}
    binary = {'income', 'gender'}

    def load(self) -> pd.DataFrame:
        path = kagglehub.dataset_download('wenruliu/adult-income-dataset')
        data = pd.read_csv(path + '/adult.csv').replace('?', np.nan)
        data['income'] = data['income'].map(lambda s: s.removesuffix('.'))
        return data

    def protected(self, continuous: bool) -> List[str]:
        p = ['gender', 'race_White', 'race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black', 'race_Other']
        return [*p, 'age'] if continuous else p


class Compas(DatasetInfo):
    classification = True
    target = 'Recidivism'
    numeric = {'Number_of_Priors'}
    binary = {'Gender', 'Misdemeanor', 'Recidivism'}

    def load(self) -> pd.DataFrame:
        path = kagglehub.dataset_download('danofer/compass')
        data = pd.read_csv(path + '/propublicaCompassRecividism_data_fairml.csv/propublica_data_for_fairml.csv')
        data = data.drop(columns='score_factor').rename(columns={
            'Two_yr_Recidivism': 'Recidivism',
            'Age_Below_TwentyFive': '<25',
            'Age_Above_FourtyFive': '>45',
            'Female': 'Gender'
        })
        # restore categorical columns to include the missing value
        categories = dict(
            Age=['<25', '>45', '25-45'],
            Race=['African_American', 'Asian', 'Hispanic', 'Native_American', 'Other', 'White']
        )
        for key, columns in categories.items():
            cat = data[columns[:-1]]
            data = data.drop(columns=columns[:-1])
            cat[columns[-1]] = 1 - cat.sum(axis=1)
            data[key] = (cat == 1).idxmax(axis=1)
        return data

    def protected(self, continuous: bool) -> List[str]:
        return [
            'Gender',
            'Race_White',
            'Race_African_American',
            'Race_Asian',
            'Race_Hispanic',
            'Race_Native_American',
            'Race_Other'
        ]


class Law(DatasetInfo):
    classification = True
    target = 'passed'
    numeric = {'lsat', 'ugpa', 'decile1', 'decile3', 'fam_inc'}
    binary = {'gender', 'parttime', 'passed'}

    def load(self) -> pd.DataFrame:
        path = kagglehub.dataset_download('danofer/law-school-admissions-bar-passage')
        data = pd.read_csv(path + '/bar_pass_prediction.csv')
        data = data[['lsat', 'ugpa', 'decile1', 'decile3', 'parttime', 'race1', 'gender', 'fam_inc', 'bar_passed']]
        return data.rename(columns={'race1': 'race', 'bar_passed': 'passed'})

    def protected(self, continuous: bool) -> List[str]:
        p = ['gender', 'race_white', 'race_black', 'race_asian', 'race_hisp', 'race_other']
        return [*p, 'fam_inc'] if continuous else p


class Student(DatasetInfo):
    classification = False
    target = 'G3'
    numeric = {
        'age',
        'Medu',
        'Fedu',
        'traveltime',
        'studytime',
        'failures',
        'famrel',
        'freetime',
        'goout',
        'Dalc',
        'Walc',
        'health',
        'absences',
        'G3'
    }
    binary = {
        'sex',
        'school',
        'address',
        'famsize',
        'Pstatus',
        'schoolsup',
        'famsup',
        'paid',
        'activities',
        'nursery',
        'higher',
        'internet',
        'romantic'
    }

    def load(self) -> pd.DataFrame:
        path = kagglehub.dataset_download('larsen0966/student-performance-data-set')
        return pd.read_csv(path + '/student-por.csv').drop(columns=['G1', 'G2'])

    def protected(self, continuous: bool) -> List[str]:
        return ['sex', 'school', 'address', 'age'] if continuous else ['sex', 'school', 'address']


class Dataset(Item):
    """Interface for a dataset used in the experiments."""

    @classmethod
    @final
    def last_edit(cls) -> str:
        return "2025-04-24 00:00:00"

    def __init__(self, name: str, continuous: bool) -> None:
        """
        :param name:
            The name of the dataset.

        :param continuous:
            Whether the protected attributes can include continuous features as well.
        """
        # handle dataset based on name
        if name == 'adult':
            benchmark = Adult()
        elif name == 'compas':
            benchmark = Compas()
        elif name == 'law':
            benchmark = Law()
        elif name == 'student':
            benchmark = Student()
        else:
            raise AssertionError(f"Unknown dataset '{name}'")
        self._continuous: bool = continuous
        self._benchmark: DatasetInfo = benchmark

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self._benchmark.__class__.__name__.lower(), continuous=self._continuous)

    @property
    def classification(self) -> bool:
        return self._benchmark.classification

    @property
    def target(self) -> str:
        """The name of the target feature."""
        return self._benchmark.target

    @property
    def features(self) -> List[str]:
        return [column for column in self._benchmark.data.columns if column != self.target]

    @property
    def protected(self) -> List[str]:
        """The names of the protected features."""
        return self._benchmark.protected(continuous=self._continuous)

    @property
    def protected_indices(self) -> List[int]:
        """The indices of the protected features within the input matrix."""
        features = {feature: index for index, feature in enumerate(self.features)}
        return [features[key] for key in self.protected]

    @property
    def x(self) -> np.ndarray:
        """The input matrix."""
        return self._benchmark.data[self.features].values

    @property
    def y(self) -> np.ndarray:
        """The output target."""
        return self._benchmark.data[[self.target]].values

    def folds(self, k: int = 1, seed: int = 0) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Returns a list of tuples <train, val> (if folds == 1, splits between train and test)."""
        data = self._benchmark.data
        if k == 1:
            stratify = data[self.target] if self.classification else None
            idx = [train_test_split(data.index, test_size=0.3, stratify=stratify, random_state=seed)]
        else:
            kf = StratifiedKFold if self.classification else KFold
            idx = kf(n_splits=k, shuffle=True, random_state=seed).split(X=data.index, y=data[self.target])
        return [(data.iloc[tr], data.iloc[ts]) for tr, ts in idx]

    @final
    def __len__(self) -> int:
        return len(self._benchmark.data)
