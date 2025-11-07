# Achieving Intersectional Algorithmic Fairness By Constructing A Maximal Correlation Latent Space

This repository provides the code for reproducing the results obtained in the paper *Achieving Intersectional Algorithmic Fairness By Constructing A Maximal Correlation Latent Space*, to be presented at ECAI 2025.

## Paper Citation

If you use this codebase, please cite the paper accordingly using the following information:
```
@inbook{Giuliani2025,
  title        = {Achieving Intersectional Algorithmic Fairness by Constructing a Maximal Correlation Latent Space},
  author       = {Giuliani,  Luca and Lombardi,  Michele},
  year         = {2025},
  month        = {oct},
  booktitle    = {ECAI 2025},
  publisher    = {IOS Press},
  doi          = {10.3233/faia251105},
  isbn         = {9781643686318},
  issn         = {1879-8314},
  url          = {http://dx.doi.org/10.3233/FAIA251105}
}
```

## Installation Setup

There are two possible ways to run the scripts in this repository.

#### Local Execution

The first is via a local Python interpreter available in the machine.
Our code was tested with `Python 3.12`, so we suggest to stick to this version if possible.
We also recommend the use of a virtual environment in order to avoid package conflicts due to specific versions.
At this point, you will need to install the requirements via:
```
pip install -r requirements.txt
```
and eventually run one of the two available scripts using the command:
```
python <baselines | gedi>.py
```
optionally passing the specific parameters.

#### Container Execution

The second way is by leveraging Docker containers.
Each python script is associated to a Docker service that can be invoked and executed in background using the command:
```
docker compose run <baseline | gedi>
```
If you do not specify the name of the script, _both_ scripts will be launched together in parallel, so we do not recommend this option.
To interact with the repository using a terminal in order to, e.g., launch scripts with custom parameters, it is possible to use the command:
```
docker compose run main -it
```
which will simply open the container in the working directory without launching any Python script, and then using the container as a local machine.

## Scripts Description

Each script contains the code to generate specific a table in the paper.

When running a script, this will automatically store the serialized results in a new folder named `results` within the project, so that they are not recomputed in case of a new execution.
It is possible to call the `clear` script/service in order to remove certain or all the obtained results, passing the specific parameters described in the script documentation.
Output figures and tables are also stored in the `results` folder.
If you would like to change the position of the folder you can specify the `-f <folder_path>` parameter when running a script locally, or exporting the environment variable `$FOLDER = <folder_path>` when using containers.

#### gedi.py

This script trains multiple neural networks using intersectional/independent gedi regularizers.
Results are shown in _Table 1_ of the paper.

#### baselines.py

This script trains multiple neural networks using different intersectional fairness regularizers.
Results are shown in _Table 2_ of the paper.

## Contacts

In case of questions about the code or the paper, do not hesitate to contact the corresponding author **Luca Giuliani** ([luca.giuliani13@unibo.it](mailto:luca.giuliani13@unibo.it)).
