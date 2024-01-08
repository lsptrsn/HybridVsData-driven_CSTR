Hybrid Model CO2 Methanation
==============================

Hybrid Model CO2 Methanation uses data and physical knowledge to create hybrid models for the CO2 methanation reaction

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── src                <- Source code for use in this project.
    │   ├── data           <- Scripts to download or generate data
    │   │   └── preprocessing.py
    │   │   └── utils_data.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling.
    │   │   └── build_features.py
    │   │
    │   ├── fit_results    <- Store fitted results.
    │   │
    │   ├── mechanistic_part <- Mechanistic parameters and calculations.
    │   │   └── balance.py
    │   │   └── mechanistic_parameter.py
    │   │   └── mechanistics.py
    │   │   └── reaction_rate.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make.
    │   │   │                 predictions
    │   │   ├── MLPR_architecture.py
    │   │   ├── score_benchmark.py
    │   │   └── train_model.py
    │   │
    │   └── path.py
    │   └── workflow.py


--------

