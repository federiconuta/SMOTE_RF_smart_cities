# SMOTE procedure with Random Forest
The repository provides a simplified version of the code used for SMOTING the data in the paper "The Technological Complexity of Global Cities: A Machine-Learning Analysis" by Nutarelli et al.

# Random Forest Classification with SMOTE and Hyperparameter Tuning

This repository contains a Python script for performing **Random Forest Classification** with **SMOTE-based data augmentation** and hyperparameter tuning using `GridSearchCV`. The main objective is to optimize model performance on imbalanced datasets by applying SMOTE (Synthetic Minority Over-sampling Technique) and tuning key hyperparameters.

## Table of Contents
1. [Overview](#overview)
2. [Functionality](#functionality)
3. [Usage](#usage)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Running the Script](#running-the-script)
7. [Output Files](#output-files)

---

### 1. Overview

This script is designed to:
- Process city-year-based data to prepare features and labels.
- Apply SMOTE for resampling imbalanced datasets.
- Train a `RandomForestClassifier` with hyperparameter tuning to improve prediction accuracy.
- Save the results for each city and year, and aggregate results to provide insights.

### 2. Functionality

The project includes the following main functionalities:

- **Data Processing**: Loads city data files, formats data by binary encoding, and splits it into training and testing sets.
- **SMOTE Application**: Applies SMOTEN resampling to address imbalanced data within the training set.
- **Hyperparameter Tuning**: Uses `GridSearchCV` to tune hyperparameters for a `RandomForestClassifier`, optimizing parameters to maximize the F1-score.
- **Parallel Processing**: Processes multiple city-year combinations in parallel to enhance computational efficiency.
- **Result Aggregation**: Aggregates results across cities to provide average hyperparameter values and key metrics.

### 3. Usage

This guide provides step-by-step instructions for inexperienced users. The user ca revolve to the folder data_csv for an example with two cities util 2010.

#### Step 1: Clone the Repository

To get started, clone this repository to your local machine:

```
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```

#### Step 2: Configure File Paths

Edit the paths in the script to match your file structure:

-`DATA_PATH:` Set to the directory where your data files are located.
-`OUTPUT_PATH:` Set to the directory where you want the results saved.

```
# In the script, replace these paths with your directories
DATA_PATH = "path/to/data_directory"
OUTPUT_PATH = "path/to/output_directory"
```

##### Step 3: Run the Script

To run the script, use the following command:
```
python rf_smote_classifier.py
```
The script will process each city and year in parallel and save the results to CSV files in the `OUTPUT_PATH` directory.

#### 4. Requirements
The script requires the following Python packages:

```
pandas (for data manipulation)
numpy (for numerical operations)
scikit-learn (for machine learning and metrics)
imblearn (for SMOTE, a part of the imbalanced-learn library)
joblib (for parallel processing)
```
Ensure these are installed by using the requirements.txt file provided.

#### 5. Installation
To set up the environment, use the following commands:

```
# Create a virtual environment (recommended but optional)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\\Scripts\\activate`
```

# Install dependencies

```
pip install -r requirements.txt
```

#### 6. Running the Script
To run the script:

Customize the list of cities and year range as desired.
Ensure data files are named consistently with the format <city>_cd_<year>_tech.csv, if we wat to analyse rome, for example the corresposing files should be saved as rome_cd_2000_tech.csv, rome_cd_2001_tech.csv,...,rome_2014.csv.
Call the main() function with the desired list of cities and years:
```
# Uncomment the following line in the script to run with actual data
main(cities=['City1', 'City2'], years=range(2000, 2009), param_grid=param_grid)
```

#### 7. Output Files
The script produces two key output files:

- results.csv: A detailed record of model metrics and optimal parameters for each city-year combination.
- avg_results.csv: Aggregated average hyperparameters and metrics across cities.

### Evaluate Random Forest Model with Averaged Hyperparameters

To evaluate a `RandomForestClassifier` using averaged hyperparameters, use the provided `evaluate_rf_with_avg_params.py` script. This script loads `avg_results.csv` to retrieve the best parameters for each city and predicts target values for 2014 based on training data from 2009.

#### Usage

1. Ensure `avg_results.csv` and relevant data files are in your specified `data_path`.
2. Run the script with:
   
   `from evaluate_rf_with_avg_params import evaluate_rf_with_avg_params`
   `evaluate_rf_with_avg_params(data_path='path/to/data_directory', output_path='path/to/output_directory')`










