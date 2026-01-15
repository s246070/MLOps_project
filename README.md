# mlops_project

The overall goal of this project is to build a machine learning pipeline capable of classifying whether an individual passenger survived the Titanic disaster based on demographic and ticket-related attributes. Using the Titanic dataset from Kaggle, the project aims not only to achieve reasonable predictive performance, but also to provide interpretable insights into which passenger characteristics were most strongly associated with survival. In particular, the project focuses on understanding how factors such as ticket class, fare, age, gender, and family relations influenced survival probabilities.

The project is implemented using a PyTorch-based framework. PyTorch is chosen due to its flexibility, clear computational graph structure, and widespread use in modern machine learning workflows. The framework is integrated into the project through a modular structure consisting of separate scripts for data preprocessing, model definition, training, evaluation, and exploratory analysis. Additionally, Weights & Biases (wandb) is used for experiment tracking, hyperparameter sweeps, and logging of metrics and visualizations.

The primary dataset used in this project is the Kaggle Titanic dataset, which contains 891 observations corresponding to individual passengers aboard the RMS Titanic. Each observation includes 12 variables describing passenger demographics, socio-economic status, and travel information, including passenger class, sex, age, fare, number of family members aboard, and port of embarkation. The target variable indicates whether the passenger survived the disaster. Prior to modeling, the data is cleaned by handling missing values, encoding categorical variables, and splitting the dataset into training and test sets.

The model used in this project is a linear classifier in the form of logistic regression. Logistic regression is particularly suitable for this task because the target variable is binary and because the model provides interpretable coefficients. These coefficients can be transformed into odds ratios, allowing direct interpretation of how each feature affects survival odds. This aligns well with the project’s analytical goal of identifying which types of passengers had the best chances of survival, rather than focusing solely on predictive accuracy.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
