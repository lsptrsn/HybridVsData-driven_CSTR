This repository contains the code and routines used in the manuscript "Questioning modeling expectations: A critical analysis of data-driven and hybrid models exemplified in $\mathrm{CO_2}$ methanation".

## Abstract

This study takes a critical look at the widely held belief that hybrid models are superior to purely data-driven models, particularly in the context of reactive systems. We put this assumption to the test by conducting a comparative analysis of four process data-driven models, including one purely data-driven model and three hybrid models. The hybrid models combine data-driven and mechanistic approaches, using data-driven submodels for specific process parts and data correction for mechanistic errors. Our investigation uses simulated data from a case study of catalytic $\mathrm{CO_2}$ methanation in a continuously stirred tank reactor. This case study serves as a representative example of a broader context in which accurate calibration of reactor models is paramount to optimizing plant operations in process design. We evaluate these models based on their accuracy, training effort, and reliability. Contrary to popular belief, our results show that hybrid models do not consistently outperform the purely data-driven model. This highlights the importance of careful model selection based on the specifics of the problem at hand. The choice between hybrid and purely data-driven models should be based on a careful evaluation of the balance between effort and potential benefit, rather than a blanket preference for one over the other.

## Repository Structure

The repository is organized into two main directories:

1. `hybrid_models_co2-methanation`: This directory contains the code to build up a data-driven model and three hybrid models as mentioned in the abstract.

2. `mech_model_co2-methanation_CSTR`: This directory contains the code to build up the mechanistic CSTR model.

Each directory contains its own README file with more detailed instructions on how to use the code within.

## Getting Started

Please refer to the README files in each directory for instructions on how to use the models.

## Dependencies

For reproducibility, we have stored all dependencies and their versions in `environment.yml`. An virtual enviornment, namely `ptx_env' can be created using `conda` by using `conda env create -f environment.yml`.

## License
This material is licensed under the MIT [LICENSE](LICENSE) and is free and provided as-is.
