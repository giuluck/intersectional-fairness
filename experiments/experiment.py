import gc
import hashlib
import itertools
import json
import logging
import os
import pickle
import shutil
import time
from abc import abstractmethod
from typing import Dict, Any, List, Iterable, Callable, Tuple, Optional
from typing import Union, Literal

import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from items import Dataset, Metric, Loss, Fairness, Model
from items import Item

PALETTE: List[str] = ['#000000', '#377eb8', '#ff7f00', '#4daf4a']
"""The color palette for plotting data."""

ITERATIONS: int = 10
"""The number of Moving Targets iterations,"""

SEED: int = 0
"""The random seed used in the experiment."""


class Experiment:
    """An experiment where a neural network is constrained so that the correlation between a protected variable and the
    target is reduced."""

    @classmethod
    def alias(cls) -> str:
        """The alias of the class of experiments."""
        return 'experiment'

    @classmethod
    def routine(cls, experiment: 'Experiment') -> Dict[str, Any]:
        """Defines the routine of the experiment.

        :param experiment:
            The signature of the experiment.

        :return:
            A dictionary containing the results of the experiment.
        """
        gc.collect()
        pl.seed_everything(SEED, workers=True)
        output = Model(indicator=experiment._indicator).run(
            dataset=experiment._dataset,
            folds=experiment._folds,
            fold=experiment._fold,
            seed=SEED
        )
        return dict(
            metric={},
            train_inputs=output.train_inputs,
            train_target=output.train_target,
            train_predictions=output.train_predictions,
            val_inputs=output.val_inputs,
            val_target=output.val_target,
            val_predictions=output.val_predictions,
            **output.additional
        )

    @classmethod
    def signatures(cls, **parameters: Any) -> Dict[Any, Dict[str, Any]]:
        """Build the signatures of the experiments that must be executed given the lists of parameters.

        :param parameters:
            A dictionary associating to each parameter name either a single value or an iterable of such.

        :return:
            A dictionary containing the signatures {key: value} paired with their output index.
            E.g., if the parameters are {p1: {a: 1, b: 2}, p2: [abc, cba], p3: 100} then we will have that:
              - the names are [p1, p2, p3]
              - the values are [(1, abc, 100), (2, abc, 100), (1, cba, 100), (2, cba, 100)]
              - the indices are [(a, abc), (b, abc), (a, cba), (b, cba)
        """
        # get the list of parameter names from the key of the parameters
        keys = parameters.keys()
        # build the experiment index and its values by iterating over the parameter values:
        #  - values are always converted into lists
        #  - indices are either ignored (non-iterable objects) or taken from either keys or values
        indices, values = [], []
        for value in parameters.values():
            if isinstance(value, Dict):
                indices.append(list(value.keys()))
                values.append(list(value.values()))
            elif isinstance(value, List):
                value = list(value)
                values.append(value)
                indices.append(value)
            else:
                values.append([value])
        # build the cartesian product of indices and values using the itertools utility
        indices = itertools.product(*indices)
        values = itertools.product(*values)
        # compute the final dictionary of signatures by pairing each index to the respective signature
        # where the signature is built by pairing together the parameter keys with the obtained values
        return {index: {k: v for k, v in zip(keys, value)} for index, value in zip(indices, values)}

    @classmethod
    def get_output_file(cls, folder: str) -> str:
        """Gets the path of the experiment output file.

        :param folder:
            The folder where results are stored and loaded.

        :return:
            The path of the experiment output file.
        """
        return os.path.join(folder, cls.alias(), 'experiments.pkl')

    @classmethod
    def get_external_folder(cls, folder: str, key: str) -> str:
        """Gets the folder where the result files of an experiment will be stored.

        :param folder:
            The folder where results are stored and loaded.

        :param key:
            The key of the experiment.

        :return:
            The folder where the result files of an experiment will be stored.
        """
        return os.path.join(folder, cls.alias(), key)

    def __init__(self,
                 folder: str,
                 dataset: Dataset,
                 indicator: Optional[str],
                 folds: int,
                 fold: int):
        """
        :param folder:
            The folder where the experiment will be stored/loaded.

        :param dataset:
            The dataset used in the experiment.

        :param indicator:
            The indicator to use as regularizer, or None for no regularizer.

        :param folds:
            The number of folds for k-fold cross-validation.

        :param fold:
            The fold that is used for training the model.
        """
        self._dataset: Dataset = dataset
        self._indicator: Optional[str] = indicator
        self._fold: int = fold
        self._folds: int = folds
        self._folder: str = folder
        self._key: Optional[str] = None
        self._built: Optional[Dict[str, Any]] = None
        self._external_results: Dict[str, Any] = {}

    def _build(self, execution_time: str, elapsed_time: float, results: Dict[str, Any]) -> None:
        """Builds the experiment with the run-specific information.

        :param execution_time:
            A timestamp which indicates when was the experiment executed.

        :param elapsed_time:
            The time (in seconds) that it took to execute the experiment.

        :param results:
            The internal results of the experiment.
        """
        assert self._built is None, "Experiment has been already built."
        self._built = dict(
            execution=execution_time,
            elapsed=elapsed_time,
            internal=results
        )

    @property
    def files(self) -> Dict[str, str]:
        """Dictionary which associates a filename to each result key. E.g., if the results have keys {'alpha', 'beta'}
        then one might indicate {'alpha': 'file_alpha', 'beta': 'file_beta'} to store each key in a different file. If
        a key is returned by the routine but is not present in this dictionary, it is stored in the main pickle file."""
        # store additional files for history/predictions in case of lagrangian dual or moving targets algorithm
        additional_files = {'history': 'history'}
        for s in range(Model.epochs):
            additional_files[f'train_prediction_{s}'] = f'pred-{s}'
            additional_files[f'val_prediction_{s}'] = f'pred-{s}'
        return dict(
            train_inputs='data',
            train_target='data',
            train_predictions='data',
            val_inputs='data',
            val_target='data',
            val_predictions='data',
            metric='metric',
            **additional_files
        )

    @property
    @abstractmethod
    def signature(self) -> Dict[str, Any]:
        """A json-compliant signature of the experiment, which must contain primitive types only."""
        return dict(
            dataset=self._dataset.configuration,
            indicator=self._indicator,
            folds=self._folds,
            fold=self._fold
        )

    @property
    def _internal_results(self) -> Dict[str, Any]:
        """The dictionary of results which are stored in the main pickle file."""
        assert self._built is not None, "The experiment has not been built yet."
        return self._built['internal']

    @property
    def execution_time(self) -> str:
        """A timestamp which indicates when was the experiment executed."""
        assert self._built is not None, "The experiment has not been built yet."
        return self._built['execution']

    @property
    def elapsed_time(self) -> float:
        """The time (in seconds) that it took to execute the experiment."""
        assert self._built is not None, "The experiment has not been built yet."
        return self._built['elapsed']

    @property
    def key(self) -> str:
        """A unique key (hash) of the experiment computed from its signature."""
        if self._key is None:
            algorithm = hashlib.sha256()
            buffer = json.dumps(self.signature, sort_keys=True).encode()
            algorithm.update(buffer)
            self._key = algorithm.hexdigest()
        return self._key

    @property
    def dump(self) -> Dict[str, Any]:
        """The dictionary dump of the experiment, containing signature and main results."""
        return dict(
            signature=self.signature,
            elapsed_time=self.elapsed_time,
            execution_time=self.execution_time,
            results=self.results(load='internal')
        )

    @property
    def folder(self) -> str:
        return self._folder

    @property
    def output_file(self) -> str:
        """The path of the experiment output file."""
        return self.get_output_file(folder=self.folder)

    @property
    def external_folder(self) -> str:
        """The folder where the result files of the experiment will be stored."""
        return self.get_external_folder(folder=self.folder, key=self.key)

    def update(self, flush: bool = True, **results: Any) -> None:
        """Updates the results of the experiment which are stored on external files.

        :param flush:
            Whether to store the results and then flush them from the cache, or to keep them cached.

        :param results:
            The results to be stored, which must be associated to an external file through the 'external' property.
        """
        # for each result:
        #  - remove it from the cache of external results if present
        #  - check that the result is assigned to an external file
        #  - build a new dictionary pairing each filename with its content (i.e., a sub-dictionary of the results)
        files = dict()
        for key, value in results.items():
            if key in self._external_results:
                self._external_results.pop(key)
            filename = self.files.get(key)
            if filename is None:
                raise RuntimeError(f'It is possible to update results in external files only, got key "{key}"')
            content = files.get(filename, {})
            content[key] = value
            files[filename] = content
        # if there is at least one external file create the folder, then for each file:
        #  - if present, load the previous results and update it with the new content
        #  - dump the result dictionary before writing to check that it is pickle-compliant
        #  - store the dumped dictionary
        folder = self.external_folder
        if len(files) > 0:
            os.makedirs(folder, exist_ok=True)
        for filename, content in files.items():
            filepath = os.path.join(folder, f'{filename}.pkl')
            if os.path.exists(filepath):
                with open(filepath, 'rb') as file:
                    previous_content = pickle.load(file=file)
                content = {**previous_content, **content}
            dump = pickle.dumps(content)
            with open(filepath, 'wb') as file:
                file.write(dump)
        # if the results must not be flushed, store them in the internal cache, then return the experiment itself
        if not flush:
            self._external_results.update(results)
        return

    def results(self, load: Literal['internal', 'cached', 'all'] = 'all') -> Dict[str, Any]:
        """Returns the results of the experiment.

        :param load:
            If 'internal', returns the internal results only.
            If 'cached', returns all the results but assigns 'None' for all those which have not been cached yet.
            If 'all', returns all the results and caches those which are had not been cached yet.

        :return:
            A dictionary containing the (partial or complete) results of the experiment.
        """
        if load == 'internal':
            external = {}
        elif load == 'cached':
            external = {key: self._external_results.get(key) for key in self.files.keys()}
        elif load == 'all':
            external = {key: self.get(name=key, cache=True) for key in self.files.keys()}
        else:
            raise AttributeError(f'Parameter "load" must be either "internal", "cached", or "all", got "{load}"')
        # return a dictionary where external results are updated with the already present ones
        return {**self._internal_results, **external}

    def free(self) -> None:
        """Empties the cache of external results."""
        self._external_results.clear()

    def get(self, name: str, cache: bool = False) -> Any:
        """Retrieves a result given its name. The result can be stored in the main pickle file or in one of the
        external files. In the first case, the value is simply returned, otherwise it the external file has not been
        loaded yet, it is loaded and optionally cached for future accesses in case 'cache' is True.

        :param name:
            The name of the result to retrieve.

        :param cache:
            Whether to cache the obtained results in case it is in an external file.

        :return:
            The value of the given result.
        """
        if name in self._internal_results:
            value = self._internal_results[name]
        elif name in self._external_results:
            value = self._external_results[name]
        else:
            # get the folder of the experiment
            value = None
            filename = self.files[name]
            filepath = os.path.join(self.external_folder, f'{filename}.pkl')
            # if it exists, load the results and update the external results dictionary if necessary
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as file:
                    content = pickle.load(file=file)
                if cache:
                    self._external_results.update(content)
                value = content.get(name)
        return value

    def __getitem__(self, item: str) -> Any:
        return self.get(name=item, cache=True)

    @classmethod
    def execute(cls,
                folder: str,
                verbose: Optional[bool] = False,
                save_time: Optional[float] = 60,
                **parameters: Any) -> dict:
        """Executes many experiment using a factorial design.

        :param folder:
            The folder where results are stored and loaded.

        :param verbose:
            If True, prints information about each experiment that is being executed.
            If False, keeps track of the progress with a progress bar.
            If None, suppresses the progress bar as well.

        :param save_time:
            The minimum amount of time (in seconds) after which to save the newly executed experiments.

        :param parameters:
            A dictionary associating to each parameter name either a single value or an iterable of such.
            In case of iterable, the values will be used to get the factorial design.

        :return:
            The list of executed experiments.
        """
        assert len(parameters) > 0, "There must be at least one parameter to define the experiment."
        # build a dictionary of output experiments and load the dictionary of already computed experiments
        experiments = {}
        dumps = cls.load(folder=folder)
        # keep track of whether there is at least a new experiment to save and the last time when they were saved
        # then get the dictionary of signatures, pair each signature {key: value} to its index in the output dictionary
        to_save = False
        last_save = time.time()
        signatures = cls.signatures(**parameters)
        iterable = tqdm(signatures.items(), desc='Fetching Experiments') if verbose is False else signatures.items()
        # iterate over every signature to execute
        for i, (index, signature) in enumerate(iterable):
            # noinspection PyArgumentList
            exp = cls(folder=folder, **signature)
            # - check whether the experiment has been already executed by looking for its key
            # - if a match is found, check that it is not outdated with respect to any of its items
            #    > if outdated, remove both its entry in the main file and its results from the external folder
            dump = dumps.get(exp.key)
            if dump is not None:
                edit = pd.Timestamp(dump['execution_time'])
                for item in signature.values():
                    if isinstance(item, Item) and pd.Timestamp(item.last_edit()) > edit:
                        dump = None
                        dumps.pop(exp.key)
                        ext_folder = exp.external_folder
                        if os.path.isdir(ext_folder):
                            shutil.rmtree(ext_folder)
                            logging.info(f'Removing experiment {exp.key} ({edit}) due to {item} ({item.last_edit()})')
                        break
            # in case there is no valid experiment retrieved (no match or outdated match):
            #  - print information if necessary
            #  - build the experiment using the private method
            #  - add the configuration dictionary to the exp_configs dictionary
            #  - if enough time has passed, store the exp_configs dictionary to the main file
            # otherwise build a new experiment instance using the information from the stored experiments
            if dump is None:
                if verbose:
                    print(flush=True)
                    print(f'Running Experiment {i + 1} of {len(signatures)}:')
                    for parameter, value in signature.items():
                        print(f'  > {parameter.upper()}: {value}')
                    print(end='', flush=True)
                to_save = True
                cls.run(experiment=exp)
                dumps[exp.key] = exp.dump
                if save_time is not None and time.time() - last_save > save_time:
                    to_save = False
                    last_save = time.time()
                    cls.store(folder=folder, dumps=dumps)
            else:
                exp._build(**{k: v for k, v in dump.items() if k != 'signature'})
            # add the exp to the output dictionary using the appropriate index
            experiments[index] = exp
        # store the experiment dumps if necessary and return the dictionary of output experiments
        if to_save:
            cls.store(folder=folder, dumps=dumps)
        return experiments

    @classmethod
    def load(cls, folder: str) -> Dict[str, Dict[str, Any]]:
        """Loads the stored experiments.

        :param folder:
            The folder where results are loaded from.

        :return:
            A dictionary {index: signature} containing the signature of each experiment indexed by their key.
        """
        # if there is already a file containing previous experiments load it, otherwise make the folder
        dumps = {}
        filepath = cls.get_output_file(folder=folder)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as file:
                dumps = pickle.load(file)
        else:
            exp_folder = os.path.dirname(filepath)
            os.makedirs(exp_folder, exist_ok=True)
        return dumps

    @classmethod
    def store(cls, folder: str, dumps: Dict[str, Dict[str, Any]]) -> None:
        """Stores the computed experiments.

        :param folder:
            The folder where results are stored.

        :param dumps:
            A dictionary {index: signature} containing the signature of each experiment indexed by their key.
        """
        # dump the experiments as a pickle file
        dump = pickle.dumps(dumps)
        # write the output file with the configurations
        filepath = cls.get_output_file(folder=folder)
        with open(filepath, 'wb') as file:
            file.write(dump)

    @classmethod
    def run(cls, experiment: 'Experiment') -> None:
        """Runs the experiment using its routine, then stores its internal and external results within the instance.

        :param experiment:
            The partial experiment instance which has not been run yet.
        """
        # runs the routine and keeps track of the execution and elapsed times
        execution_time = str(pd.Timestamp.now())
        elapsed_time = time.time()
        results = cls.routine(experiment=experiment)
        elapsed_time = time.time() - elapsed_time
        # splits the results into internal (stored directly in the main file) and external (stored as separate pickles)
        split_results = dict(internal={}, external={})
        for k, v in results.items():
            split_results['external' if k in experiment.files else 'internal'][k] = v
        # builds the experiment instance using the internal results, then updates the external ones
        experiment._build(execution_time=execution_time, elapsed_time=elapsed_time, results=split_results['internal'])
        experiment.update(flush=True, **split_results['external'])

    @classmethod
    def inspection(cls, folder: str, show: bool = True, export: Iterable[Literal['csv', 'json']] = ()) -> pd.DataFrame:
        """Inspects the signatures of the run experiments.

        :param folder:
            The folder where results are stored.

        :param show:
            Whether to print the dataframe of retrieved signatures on screen.

        :param export:
            The extension of the export files ('csv' or 'json') containing the signatures of the experiments.0
        """

        # recursively flatten a dictionary using tuples as keys
        def flatten(dictionary: Dict[str, Any], parent: Iterable[str] = ()) -> Dict[Tuple[str, ...], Any]:
            output = dict()
            parent = tuple(parent)
            for key, value in dictionary.items():
                new_parent = (*parent, key)
                if isinstance(value, dict):
                    output = {**output, **flatten(dictionary=value, parent=new_parent)}
                else:
                    output[new_parent] = value
            return output

        # retrieve and flatten the signatures of the experiments (if they exist)
        filepath = os.path.join(folder, cls.alias())
        if not os.path.isdir(filepath):
            return pd.DataFrame()
        signatures = [flatten({'key': key, **exp['signature']}) for key, exp in cls.load(folder=folder).items()]
        # if a json export is required, dump a json file with the signatures
        # use ':' to create a single string key by joining the tuples
        if 'json' in export:
            filepath = os.path.join(folder, f'{cls.alias()}.json')
            json.dump([{':'.join(k): v for k, v in s.items()} for s in signatures], fp=open(filepath, 'w'), indent=2)
        # set the index, then use the same tuple length for each column to build a multi-column
        signatures = pd.DataFrame(signatures).set_index(('key',))
        signatures.index.name = None
        length = max([len(column) for column in signatures.columns])
        columns = [column + ('',) * (length - len(column)) for column in signatures.columns]
        signatures.columns = pd.MultiIndex.from_tuples(columns)
        # if a csv export is required, export the built dataframe
        if 'csv' in export:
            filepath = os.path.join(folder, f'{cls.alias()}.csv')
            signatures.to_csv(filepath)
        # if a print is required, print the whole dataset (use the 'to_string' method to avoid ellipses)
        if show:
            print(signatures.to_string(sparsify=False))
        return signatures

    @classmethod
    def clear(cls,
              *conditions: str,
              folder: str,
              older: Union[None, str, pd.Timestamp] = None,
              force: bool = False,
              verbose: bool = True) -> None:
        """Clears the results based on some given conditions.

        :param folder:
            The folder where results are stored.

        :param conditions:
            Strings of type "<item>[:<subkey>:<subsubkey>:...] = <value>" where <item> is the name of the item to check,
            potentially adding various subkeys separated by ':' to refine the search, and <value> is a single value
            which must match the item/subkey that is found in the signature of the experiment.

        :param older:
            A datetime element before which all the experiments will be removed (ignores the condition if None).

        :param force:
            Whether to remove all the experiment automatically or ask for confirmation.

        :param verbose:
            Whether to print additional information about the retrieval.
        """
        filepath = os.path.join(folder, cls.alias())
        if not os.path.isdir(filepath):
            return
        # retrieve the experiment from the experiment file, and retrieve all the names of the subfolders related to the
        # experiment (the folders are used to check if there are some orphan external files to be deleted)
        folders = {key: None for key in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, key))}
        experiments = {**folders, **cls.load(folder=folder)}
        if verbose:
            print(f"Retrieved {len(experiments)} experiments.")
        # create a dictionary of output experiments, and a list of folders to be removed
        outputs = {}
        folders = []
        older = None if older is None else pd.Timestamp(older)
        # iterate over each subfolder
        for key, dump in experiments.items():
            # retrieve the experiment which corresponds to the key
            ext_folder = cls.get_external_folder(folder=folder, key=key)
            # if the key has to be removed append its folder to the list, otherwise include it in the dictionary
            if cls.remove(dump=dump, older=older, conditions=conditions):
                folders.append(ext_folder)
            else:
                outputs[key] = dump
        # if not force ask for confirmation (if the answer is not 'y' or 'yes', abort)
        # otherwise print a message if necessary
        if not force:
            msg = f"\nAre you sure you want to remove {len(experiments) - len(outputs)} experiments, "
            msg += f"leaving {len(outputs)} experiments left? (Y/N) "
            choice = input(msg)
            if choice.lower() not in ['y', 'yes']:
                if verbose:
                    print(f"\nClearing procedure aborted\n")
                return
            elif verbose:
                print(f"\nClearing procedure started\n")
        elif verbose:
            print(f"Removing {len(experiments) - len(outputs)} ({len(outputs)} left)\n")
        # remove all the subfolders of the deleted experiments, then store the output file
        for ext_folder in tqdm(folders, desc="Removing Folders"):
            if os.path.isdir(ext_folder):
                shutil.rmtree(ext_folder)
        cls.store(folder=folder, dumps=outputs)

    @classmethod
    def remove(cls, dump: Optional[Dict[str, Any]], older: Optional[pd.Timestamp], conditions: Iterable[str]) -> bool:
        """Decides whether to remove a dumped result based on a set of conditions and an edit time.

        :param dump:
            The signature dump of the given experiment.

        :param older:
            A datetime element before which all the experiments will be removed (ignores the condition if None).

        :param conditions:
            Strings of type "<item>[:<subkey>:<subsubkey>:...] = <value>" where <item> is the name of the item to check,
            potentially adding various subkeys separated by ':' to refine the search, and <value> is a single value
            which must match the item/subkey that is found in the signature of the experiment.
        """
        # remove a key if:
        #  - there is no associated run (i.e., the folder is orphan)
        #  - the run is too old and all the conditions are true
        if dump is None:
            return True
        if older is not None and pd.Timestamp(dump['execution_time']) > older:
            return False
        signature = dump['signature']
        for condition in conditions:
            # split between <item> and <value> using the '=' symbol
            item, value = condition.split('=')
            item = str(item).strip()
            value = str(value).strip()
            # find the keys and subkeys of the items by splitting using the ':' symbol
            # then browse through the signature using the keys/subkeys
            keys = item.split(':')
            parameter = signature
            for key in keys:
                if isinstance(parameter, dict) and key in parameter:
                    parameter = parameter[key]
                else:
                    return False
            # compare the string version of the retrieved parameter with the value
            if str(parameter) != value:
                return False
        return True

    @staticmethod
    def figures(folder: str = 'results', extensions: Iterable[str] = ('png',), plot: bool = False):
        sns.set(context='poster', style='white', font_scale=1)
        figures = {}
        for dim in [1, 3]:
            idx = 0
            previous = []
            graph = nx.Graph()
            fig = plt.figure(figsize=(21, 7), tight_layout=True)
            for layer, units in enumerate([dim, 16, 16, 8, 1]):
                nodes = list(range(idx, idx + units))
                graph.add_nodes_from(nodes, layer=layer)
                graph.add_edges_from([(i, j) for i in previous for j in nodes])
                previous = nodes
                idx += units
            pos = nx.multipartite_layout(graph, subset_key='layer')
            nx.draw_networkx_nodes(graph, pos=pos, node_size=500, node_color='#FFF', edgecolors='#000', ax=fig.gca())
            nx.draw_networkx_edges(graph, pos=pos, edge_color='#000', ax=fig.gca())
            figures[dim] = fig
        for extension in extensions:
            for name, fig in figures.items():
                fig.savefig(os.path.join(folder, f'network_{name}.{extension}'), bbox_inches='tight')
        if plot:
            for name, fig in figures.items():
                fig.show()
        for name, fig in figures.items():
            plt.close(fig)

    @staticmethod
    def gedi(datasets: Iterable[str] = ('student', 'compas', 'law', 'adult'),
             indicators: Iterable[str] = ('int', 'ind'),
             folds: int = 5,
             folder: str = 'results',
             extensions: Iterable[str] = ('png', 'csv'),
             plot: bool = False):
        aliases = {None: '//', 'int': 'Inter', 'ind': 'Indep'}
        datasets = {dataset: Dataset(name=dataset, continuous=True) for dataset in datasets}
        indicators = {aliases[ind]: ind for ind in [None, *indicators]}

        def configuration(_ds, _al, _fl):
            d = datasets[_ds]
            p = d.protected_indices
            # return a list of metrics for loss and gedi
            return dict(Dataset=_ds, regularizer=_al, fold=_fl), [
                Loss(classification=d.classification, name='Loss'),
                Fairness(indicator='int', protected=p, scale=d, name='Inter'),
                Fairness(indicator='ind', protected=p, scale=d, name='Indep')
            ]

        return Experiment._experiment(
            alias='gedi',
            configuration=configuration,
            datasets=datasets,
            indicators=indicators,
            folds=folds,
            folder=folder,
            extensions=extensions,
            plot=plot
        )

    @staticmethod
    def baselines(datasets: Iterable[str] = ('compas', 'law', 'adult'),
                  indicators: Iterable[str] = ('int', 'edf', 'spsf'),
                  folds: int = 5,
                  folder: str = 'results',
                  extensions: Iterable[str] = ('png', 'csv'),
                  plot: bool = False):
        aliases = {None: '//', 'int': 'GeDI', 'edf': 'EDF', 'spsf': 'SPSF'}
        datasets = {dataset: Dataset(name=dataset, continuous=False) for dataset in datasets}
        indicators = {aliases[ind]: ind for ind in [None, *indicators]}

        def configuration(_ds, _al, _fl):
            d = datasets[_ds]
            p = d.protected_indices
            # return a list of metrics for loss and fairness indicators
            return dict(Dataset=_ds, regularizer=_al, fold=_fl), [
                Loss(classification=d.classification, name='Loss'),
                Fairness(indicator='int', protected=p, scale=d, name='GeDI'),
                Fairness(indicator='edf', protected=p, scale=d, name='EDF'),
                Fairness(indicator='spsf', protected=p, scale=d, name='SPSF')
            ]

        return Experiment._experiment(
            alias='baselines',
            configuration=configuration,
            datasets=datasets,
            indicators=indicators,
            folds=folds,
            folder=folder,
            extensions=extensions,
            plot=plot
        )

    @staticmethod
    def _experiment(alias: str,
                    configuration: Callable[..., Tuple[Dict[str, Any], Iterable[Metric]]],
                    datasets: Dict[str, Dataset],
                    indicators: Dict[str, str],
                    folds: int,
                    folder: str,
                    extensions: Iterable[str],
                    plot: bool):
        assert len(datasets) > 0, "Please pass at least one dataset"
        sns.set(context='poster', style='whitegrid', font_scale=1)
        # +------------------------------------------------------------------------------------------------------------+
        # |                                              RUN EXPERIMENTS                                               |
        # +------------------------------------------------------------------------------------------------------------+
        metrics = []
        figures = {}
        experiments = {}
        # iterate over dataset and batches
        for name, dataset in datasets.items():
            # use dictionaries for dataset to retrieve correct configuration
            sub_experiments = Experiment.execute(
                folder=folder,
                save_time=0,
                verbose=True,
                dataset={name: dataset},
                indicator=indicators,
                fold=list(range(folds)),
                folds=folds
            )
            experiments.update(sub_experiments)
            # +--------------------------------------------------------------------------------------------------------+
            # |                                             FETCH METRICS                                              |
            # +--------------------------------------------------------------------------------------------------------+
            results = []
            for i, (index, exp) in enumerate(tqdm(sub_experiments.items(), desc='Fetching Metrics')):
                # retrieve inputs
                xtr = exp['train_inputs']
                ytr = exp['train_target']
                xvl = exp['val_inputs']
                yvl = exp['val_target']
                history = exp['history']
                # compute indicators for each step
                info, metrics = configuration(*index)
                outputs = {
                    **{f'train::{mtr.name}': history.get(f'train::{mtr.name}', []) for mtr in metrics},
                    **{f'val::{mtr.name}': history.get(f'val::{mtr.name}', []) for mtr in metrics},
                }
                # if the indicators are already pre-computed, load them
                if np.all([len(v) == len(history['step']) for v in outputs.values()]):
                    df = pd.DataFrame(outputs).melt()
                    df['split'] = df['variable'].map(lambda v: v.split('::')[0].title())
                    df['metric'] = df['variable'].map(lambda v: v.split('::')[1])
                    df['step'] = list(history['step']) * len(outputs)
                    for key, value in info.items():
                        df[key] = value
                    df = df.drop(columns='variable').to_dict(orient='records')
                    results.extend(df)
                # otherwise, compute and re-serialize them
                else:
                    for step in history['step']:
                        info['step'] = step
                        pvl = exp[f'val_prediction_{step}']
                        ptr = exp[f'train_prediction_{step}']
                        for mtr in metrics:
                            for split, (x, y, p) in zip(['train', 'val'], [(xtr, ytr, ptr), (xvl, yvl, pvl)]):
                                # if there is already a value use it, otherwise compute the metric and append the value
                                output_list = outputs[f'{split}::{mtr.name}']
                                if len(output_list) > step:
                                    value = output_list[step]
                                else:
                                    value = mtr(x=x, y=y, p=p)
                                    output_list.append(value)
                                results.append({**info, 'metric': mtr.name, 'split': split.title(), 'value': value})
                    exp.update(history={**history, **outputs})
            metrics = [metric.name for metric in metrics]
            results = pd.DataFrame(results)
            # +--------------------------------------------------------------------------------------------------------+
            # |                                             PLOT HISTORY                                               |
            # +--------------------------------------------------------------------------------------------------------+
            palette = PALETTE[:len(results['regularizer'].unique())]
            col = len(metrics) + 1
            epochs = Model.epochs
            figures[name], axes = plt.subplots(2, col, figsize=(5 * col, 8), tight_layout=True)
            for i, split in enumerate(['Train', 'Val']):
                for j, metric in enumerate(metrics):
                    j += 1
                    data = results[np.logical_and(results['split'] == split, results['metric'] == metric)]
                    sns.lineplot(
                        data=data,
                        x='step',
                        y='value',
                        estimator='mean',
                        errorbar='sd',
                        linewidth=2,
                        hue='regularizer',
                        style='regularizer',
                        palette=palette,
                        ax=axes[i, j]
                    )
                    title = metric if metric == 'Loss' else f'% {metric}'
                    axes[i, j].set_title(f"{title} - {split.title()}", pad=10)
                    axes[i, j].get_legend().remove()
                    axes[i, j].set_xticks([0, (epochs - 1) // 2, epochs - 1], labels=[1, epochs // 2, epochs])
                    axes[i, j].set_xlim([0, epochs - 1])
                    axes[i, j].set_xlabel(None)
                    axes[i, j].set_ylabel(None)
                    if i == 1:
                        lb = min([axes[i, j].get_ylim()[0] for i in [0, 1]]) if metric == 'Loss' else 0
                        ub = max([axes[i, j].get_ylim()[1] for i in [0, 1]])
                        axes[0, j].set_ylim((lb, ub))
                        axes[1, j].set_ylim((lb, ub))
            # get and plot times
            data = pd.DataFrame([{
                'regularizer': rg,
                'fold': fl,
                'time': experiment.elapsed_time
            } for (_, rg, fl), experiment in sub_experiments.items()])
            sns.barplot(
                data=data,
                x='regularizer',
                y='time',
                estimator='mean',
                errorbar='sd',
                linewidth=2,
                hue='regularizer',
                palette=palette,
                ax=axes[1, 0]
            )
            axes[1, 0].set_title('Time (s)', pad=10)
            axes[1, 0].set_xlabel(None)
            axes[1, 0].set_ylabel(None)
            # plot legend
            handles, labels = axes[0, 1].get_legend_handles_labels()
            axes[0, 0].legend(handles, labels, title='REGULARIZER', loc='center left', labelspacing=1.2, frameon=False)
            axes[0, 0].spines['top'].set_visible(False)
            axes[0, 0].spines['right'].set_visible(False)
            axes[0, 0].spines['bottom'].set_visible(False)
            axes[0, 0].spines['left'].set_visible(False)
            axes[0, 0].set_xticks([])
            axes[0, 0].set_yticks([])
        # +------------------------------------------------------------------------------------------------------------+
        # |                                               PRINT OUTPUTS                                                |
        # +------------------------------------------------------------------------------------------------------------+
        outputs = []
        # retrieve results
        for (ds, rg, fl), experiment in experiments.items():
            configuration = dict(Dataset=ds, Regularizer=rg)
            outputs.append({**configuration, 'split': 'train', 'metric': 'Time', 'value': experiment.elapsed_time})
            for i, split in enumerate(['train', 'val']):
                for j, metric in enumerate(metrics):
                    value = experiment['history'][f'{split}::{metric}'][Model.epochs - 1]
                    outputs.append({**configuration, 'split': split, 'metric': metric, 'value': value})
        outputs = pd.DataFrame(outputs).groupby(
            by=['Dataset', 'Regularizer', 'split', 'metric'],
            as_index=False
        ).agg(['mean', 'std'])
        outputs.columns = ['Dataset', 'regularizer', 'split', 'metric', 'mean', 'std']
        # build tex file
        tex = outputs.copy()
        text = []
        for _, r in tex.iterrows():
            scale = 1 if r['metric'] == 'Time' else 100
            form = '02.2f' if r['metric'] == 'Loss' else '02.0f'
            mean = f"{scale * r['mean']:{form}}"
            std = f"{scale * r['std']:{form}}"
            text.append(f'{mean} ± {std}')
        tex['text'] = text
        tex = tex.pivot(index=['Dataset', 'regularizer'], columns=['metric', 'split'], values='text')
        columns, header, subheader = '\\begin{tabular}{c', '\\textbf{Regularizer}', ''
        for metric in metrics:
            metric = 'Loss ($\\times 10^2$)' if metric == 'Loss' else f'\\% {metric}'
            header += ' & \\multicolumn{2}{c|}{\\textbf{' + metric + '}}'
            subheader += ' & \\textbf{train} & \\textbf{val}'
            columns += '|cc'
        lines = [columns + '|c}', '\\toprule', header + ' & \\textbf{Time (s)}\\\\', subheader + ' & \\\\']
        length = str(2 * len(metrics) + 2)
        for ds, group in tex.reset_index().groupby('Dataset', as_index=False):
            info = '\\textbf{Dataset}: \\begin{minipage}[b]{22pt}' + ds.title() + '\\end{minipage}'
            info += '(\\# ' + str(len(datasets[ds])) + ') \\qquad\n\\textbf{Protected}:'
            lines += ['\\midrule', '\\multicolumn{' + length + '}{l}{\n' + info + '\n}\\\\', '\\midrule']
            columns = [(m, s) for m in metrics for s in ['train', 'val']]
            group = group[[('regularizer', '')] + columns + [('Time', 'train')]]
            lines += group.to_latex(header=False, index=False).split('\n')[3:-3]
        lines += ['\\bottomrule', '\\end{tabular}']
        tex = '\n'.join(lines)
        # +--------------------------------------------------------------------------------------------------------+
        # |                                             STORE RESULTS                                              |
        # +--------------------------------------------------------------------------------------------------------+
        for extension in extensions:
            if extension == 'tex':
                file = open(os.path.join(folder, f'{alias}.tex'), 'w')
                file.writelines(tex)
                file.close()
            elif extension == 'csv':
                file = os.path.join(folder, f'{alias}.csv')
                outputs.to_csv(file, header=True, index=False, float_format=lambda v: f"{v:.2f}")
            else:
                for name, fig in figures.items():
                    fig.savefig(os.path.join(folder, f'{alias}_{name}.{extension}'), bbox_inches='tight')
        if plot:
            for name, fig in figures.items():
                fig.suptitle(f"Learning History for {name.title()}")
                fig.show()
        for name, fig in figures.items():
            plt.close(fig)
