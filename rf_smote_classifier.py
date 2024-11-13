{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # rf_smote_classifier.py\
\
import pandas as pd\
import numpy as np\
from sklearn.metrics import accuracy_score, precision_score, f1_score, matthews_corrcoef, average_precision_score\
from sklearn.model_selection import GridSearchCV, train_test_split\
from sklearn.ensemble import RandomForestClassifier\
from joblib import Parallel, delayed\
from imblearn.over_sampling import SMOTEN\
from sklearn.preprocessing import LabelEncoder\
import os\
\
# Initialize paths\
DATA_PATH = "path/to/data_directory"  # Customize this path\
OUTPUT_PATH = "path/to/output_directory"  # Customize this path\
\
# Other code sections...\
# Paste the code here from the cleaned-up script provided\
\
# Define hyperparameter grid\
param_grid = \{\
    'n_estimators': [50, 100, 200],\
    'max_depth': [10, 20, 30, None],\
    'min_samples_split': [2, 5, 10],\
    'min_samples_leaf': [1, 2, 4],\
    'bootstrap': [True, False],\
    'class_weight': ['balanced']\
\}\
\
if __name__ == "__main__":\
    # List cities and years as needed\
    main(cities=['City1', 'City2'], years=range(2000, 2009), param_grid=param_grid)\
}