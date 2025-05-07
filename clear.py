import argparse
import os

from experiments import Experiment

EXPERIMENTS = {exp.alias(): exp for exp in Experiment.__subclasses__()}

# build argument parser
parser = argparse.ArgumentParser(description='Clears the results in the experiment files')
parser.add_argument(
    '-f',
    '--folder',
    type=str,
    default='results',
    help='the path where to search and store the results and the exports'
)
parser.add_argument(
    '-e',
    '--experiments',
    type=str,
    nargs='+',
    default=EXPERIMENTS.keys(),
    choices=EXPERIMENTS.keys(),
    help='the name of the experiment (or list of such) to clear'
)
parser.add_argument(
    '-c',
    '--conditions',
    type=str,
    nargs='*',
    default=[],
    help='strings of type "<item>[:<subkey>:<subsubkey>:...] = <value>" where <item> is the name of the item to check, '
         'potentially adding various subkeys separated by : to refine the search, and <value> is a single value which '
         'must match the item/subkey that is found in the signature of the run'
)
parser.add_argument(
    '-o',
    '--older',
    type=str,
    nargs='?',
    help='a datetime element before which all the runs will be removed (ignores the condition if None)'
)
parser.add_argument(
    '--force',
    action='store_true',
    help='clears everything without asking for confirmation at the end'
)
parser.add_argument(
    '--verbose',
    action='store_true',
    help='print additional information about the retrieval'
)
parser.add_argument(
    '--exports',
    action='store_true',
    help='clears only the export files while keeping the experiment results (all the other parameters are ignored)'
)

# parse arguments and decide what to clear
args = parser.parse_args().__dict__
folder = args.pop('folder')
# clear all the exports (i.e., those files that are not a subfolder or in a subfolder)
print('Starting exports clearing procedure...')
if os.path.isdir(folder):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if os.path.isfile(path):
            os.remove(path)
# run the clearing procedure for all the experiments provided that the export flag was not passed
if args.pop('exports') is False:
    print('Starting experiments clearing procedure...')
    for k, v in args.items():
        print('  >', k, '-->', v)
    print()
    conditions = args.pop('conditions')
    for exp in args.pop('experiments'):
        print(f'{exp.upper()} EXPERIMENT:')
        EXPERIMENTS[exp].clear(folder=folder, *conditions, **args)
        print()
