# Machine Learning based Intrusion Detection System for IoT

## Overview/Description
This project aims to develop a machine learning-based intrusion detection system (IDS) tailored for Internet of Things (IoT) environments. It will leverage network traffic data and system logs to identify various cyber-attacks, contributing to the security and resilience of IoT infrastructures.

## Directory Structure
The project is organized into the following main directories:

*   `data/`: Contains datasets used for training, testing, and evaluation.
    *   `data/raw/ton_iot/`: Intended for the raw ToN_IoT dataset.
*   `docs/`: Contains documentation files, including dataset summaries and design choices.
*   `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA), experimentation, and visualization.
*   `src/`: Contains all source code for the project.
    *   `src/preprocessing/`: Scripts for data cleaning, transformation, and feature engineering.
    *   `src/models/`: Implementations of machine learning models.
    *   `src/training/`: Scripts to train and evaluate models.
    *   `src/evaluation/`: Scripts and utilities for model performance assessment.
    *   `src/utils/`: Utility functions and helper scripts.
*   `tests/`: Contains unit tests for the source code.

## Setup
The project primarily uses Python 3.x. Key dependencies include:

*   pandas
*   scikit-learn
*   numpy
*   pytest (for running tests)

A `requirements.txt` file will be added in the future for easier dependency management.

## How to Run

### 1. Data Exploration Notebook (`notebooks/01_initial_data_exploration.ipynb`)

**a. Download the ToN_IoT Dataset:**
   *   The notebook `notebooks/01_initial_data_exploration.ipynb` is designed to work with the "Train_Test_Network.csv" file from the ToN_IoT dataset.
   *   **You need to download the ToN_IoT dataset from its official source.** Search for "ToN_IoT dataset UNSW" or visit the UNSW Canberra Cyber website for download links.
   *   Once downloaded, locate the `Train_Test_Network.csv` file (it's usually within a folder like `Train_Test_datasets` or similar in the downloaded archive).
   *   Place this `Train_Test_Network.csv` file into the `data/raw/ton_iot/` directory within this project structure. The notebook expects the file at `data/raw/ton_iot/Train_Test_Network.csv`.

**b. Run the Jupyter Notebook:**
   *   Ensure you have Jupyter Notebook or JupyterLab installed (`pip install notebook` or `pip install jupyterlab`).
   *   Navigate to the project's root directory in your terminal.
   *   Launch Jupyter: `jupyter notebook` or `jupyter lab`.
   *   Open the `notebooks/` directory in the Jupyter interface and click on `01_initial_data_exploration.ipynb` to open and run the cells.

### 2. Training/Evaluation Script (`src/training/train_evaluate_baseline.py`)

This script demonstrates a baseline model training and evaluation pipeline using *dummy data*. Therefore, no dataset download is required to run this specific script.

*   Navigate to the project's root directory in your terminal.
*   Run the script using Python:
    ```bash
    python src/training/train_evaluate_baseline.py
    ```
    This will execute the script, which will create dummy data, train a simple Decision Tree model, and print evaluation metrics to the console.

## Unit Tests
Unit tests are located in the `tests/` directory and use the `pytest` framework.

*   Ensure `pytest` is installed (`pip install pytest`).
*   Navigate to the project's root directory in your terminal.
*   Run the following command to discover and execute all tests:
    ```bash
    python -m pytest
    ```
    or simply:
    ```bash
    pytest
    ```
