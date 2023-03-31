# EPP_final_project_sbp

[![image](https://img.shields.io/github/actions/workflow/status/Fotfake1/epp_final_project_sbp/main.yml?branch=main)](https://github.com/Fotfake1/epp_final_project_sbp/actions?query=branch%3Amain)
[![image](https://codecov.io/gh/Fotfake1/epp_final_project_sbp/branch/main/graph/badge.svg)](https://codecov.io/gh/Fotfake1/epp_final_project_sbp)

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Fotfake1/epp_final_project_sbp/main.svg)](https://results.pre-commit.ci/latest/github/Fotfake1/epp_final_project_sbp/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Usage

To get started, create and activate the environment with

```console
$ conda/mamba env create
$ conda activate epp_final_project_sbp
```

To build the project, type

```console
$ pytask
```

## Prerequisites: Webscraping:

For the webscraper to be functional, one has to have Chrome installed on his / her
machine. The environment is tested only on mac computers, due to the lack of windows
computers in my reach. If the webscraper does not work on your machine, I provide a
downloadable file
[Download](https://drive.google.com/file/d/1798GSbetZ_XT9EQNu5lK6quG3FtoDwiq/view?usp=share_link),
which than can be put in the src/data folder.

## Remarks

### General:

To get a first glance on the project, I suggest increasing the variable
INITIAL_TRAINING_DAYS to 1000, the variable MAX_NEIGHBORS_KNN to 5, the variable
N_BOOTSTRAP_ITERATIONS_RF to 100. In the current state, the program runs for roughly 6
hours in a macbook with M1 chip. Masking the simulation task can also drastically reduce
the runtime, but from my point of view the simulation task is the beauty of the program,
since it takes the computed algorithms to a "real world" example.

### Project structure:

The project is divided into the following mental parts, which can be thought of
separately and which are built on each other: Webscraping, data management, feature
creation, model built, model selection and simulation. Each of the mental building
blocks is located in a separately folder, excluded the model built and model selection,
which are placed together. The results can be generated through the pytask command,
plots are placed in the generated .pdf file.

### Simulation:

The simulation is at the current state not parallelized, due to time constraints.
Because of this issue, the runtime is high. All other parts run in a reasonable
timeframe.

Remarks about the outcomes: The profits in the simulation look promising at first
glance. My thoughts on that are the following ones: Even though, the machine learning
models work quite well in this context, a good portion of the profits are due to
odd-differences between individual sports-betting firms.

Remarks about next steps: I was quite happy with the intermediate final result I have
right now. Even though, I want to built on the webscraper, in order to include a second
source of information, possibly more in deep game information as well as current odds,
in order to make predictions on games in the near future. On top of that, I want to add
some other machine learning algorithms, namely support vector machines and improve on
the optimization of the logistic regression and random forests.

#### Project Specifics file explanations:

In the following part, one can find explanations for the global variables, stored in the
project_specific.yaml file in the src/data folder.

- INFORMATION_SCRAPING: Contains the url and other information, needed for the
  web-scraper.
- years: The years you wish to scrape - by default 2011 - 2023 but can be anything from
  2005 to now.
- LEAGUES: The leagues to be included. You can take just one league to make the project
  leaner, right now, the four most important european soccer leagues are included.
- MODELS: Which models to include - right now a random forest, a logistic regression and
  a K-nearest-neighbour model are included
- FEATURES_CREATED: A list, which contains all features, which are created by the
  program
- NOT_KNOWN_ON_GAMEDAY: A list, which contains all columns, which are not known on
  gameday. If new data is scraped, this list needs to be updated.
- ODD_FEATURES: A list of all odds scraped
- CONSIDERED_FEATURES: A list of all features included in the program
- CATEGORICAL_FEATURES: A list of all categorical features
- INTEGER_FEATURES: A list of all integer valued features
- LABELS: A list of labels for the plotting relevant
- CORR_PLOT_VARIABLES: All variables, which should be included in the correlation plots
- BETTING_STRATEGIES: All strategies, which are considered after computing the outcomes
  of games
- INITIAL_TRAINING_DAYS: The number of days, which are the initial training data in the
  simulation. Can not be lower than 1100. Increasing this number reduces the runtime of
  the project.
- TRAIN_SHARE: The relative share of the observations, used as a training sample.
- MIN_FEAT_LOG_REG: The minimum number of features, which should be included in the
  logistic regression
- MAX_DEPTH_OF_TREE_RF: Maximum depth of an individual classification tree in the random
  forest algorithm. Lower this number can reduce runtime.
- N_BOOTSTRAP_ITERATIONS_RF: The number of individual trees, considered for the random
  forest algorithm. Reducing this number can reduce the runtime, but worsen the
  performance of the random forest algorithm.
- N_SPLIT_CV: Number of splits during the time series cross validation. Increasing this
  number increases the runtime but decreases the bias of the resulting average
  classification error computed for the algorithms.
- MAX_NEIGHBORS_KNN: The maximum number of k-nearest-neighbours. Increasing this number
  can significantly increase the runtime.

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).
