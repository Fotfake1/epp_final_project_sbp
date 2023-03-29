# EPP_final_project_sbp

[![image](https://img.shields.io/github/actions/workflow/status/Fotfake1/epp_final_project_sbp/main.yml?branch=main)](https://github.com/Fotfake1/epp_final_project_sbp/actions?query=branch%3Amain) [![image](https://codecov.io/gh/Fotfake1/epp_final_project_sbp/branch/main/graph/badge.svg)](https://codecov.io/gh/Fotfake1/epp_final_project_sbp)

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

## Remarks

### Project structure: 
The project is divided into the following mental parts, which can be seen seperately and which built on each other: Webscraping, data management, feature creation, model built, model selection and simulation. Each of the mental building blocks is located in a seperate folder, excluded the model built and model selection, which are placed together. The results can be generated through the pytask command, plots are placed in the generated .pdf filde

### Simulation: 
The simulation is at the current state not parallelized, due to time constraints. Because of this issue, the runtime is high. All other parts run in a reasonable timeframe. 

Remarks about the outcomes: 
The profits in the simulation look promising at first glance. My thoughts on that are the following ones: Even though, the machine learning models work quite well in this context, a good portion of the profits are due to odd-differences between individual sports-betting firms. 

Remarks about next steps: 
I was quite happy with the intermediate final result I have right now. Even though, I want to built on the webscraper, in order to include a second source of information, possibly more in deep game information as well as current odds, in order to make predictions on games in the near future. On top of that, I want to add some other machine learning algorithms, namely support vector machines and improve on the optimization of the logistic regression and random forests. 


## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).
