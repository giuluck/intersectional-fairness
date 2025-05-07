import argparse
import logging
import warnings

from experiments import Experiment

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*GPU available but not used.*")
for name in ["lightning_fabric", "pytorch_lightning.utilities.rank_zero", "pytorch_lightning.accelerators.cuda"]:
    log = logging.getLogger(name)
    log.propagate = False
    log.setLevel(logging.ERROR)

# build argument parser
parser = argparse.ArgumentParser(description='Train multiple neural networks using different fairness regularizers')
parser.add_argument(
    '-f',
    '--folder',
    type=str,
    default='results',
    help='the path where to search and store the results and the exports'
)
parser.add_argument(
    '-d',
    '--datasets',
    type=str,
    nargs='+',
    choices=['compas', 'law', 'adult'],
    default=['compas', 'law', 'adult'],
    help='the datasets on which to run the experiment'
)
parser.add_argument(
    '-i',
    '--indicators',
    type=str,
    nargs='*',
    choices=['int', 'edf', 'spsf'],
    default=['int', 'edf', 'spsf'],
    help='the indicators used as regularizers'
)
parser.add_argument(
    '-k',
    '--folds',
    type=int,
    default=5,
    help='the number of folds to be used for cross-validation'
)
parser.add_argument(
    '-e',
    '--extensions',
    type=str,
    nargs='*',
    default=['png', 'csv'],
    help='the extensions of the files to save'
)
parser.add_argument(
    '--plot',
    action='store_true',
    help='whether to plot the results'
)

# parse arguments, build experiments, then export the results
args = parser.parse_args().__dict__
print("Starting experiment 'baselines'...")
for k, v in args.items():
    print('  >', k, '-->', v)
print()
Experiment.baselines(**args)
