{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import pandas as pd\
import numpy as np\
from sklearn.ensemble import RandomForestClassifier\
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score\
from sklearn.preprocessing import StandardScaler, LabelEncoder\
from imblearn.over_sampling import SMOTEN\
import os\
\
def evaluate_rf_with_avg_params(data_path, output_path, max_length=639):\
    """\
    Evaluates a RandomForestClassifier using averaged hyperparameters per city\
    and saves true, predicted values and probabilities for each city.\
\
    Parameters:\
    - data_path (str): Path to the directory containing data files and avg_results.csv.\
    - output_path (str): Path to save the output CSV files.\
    - max_length (int): Maximum length of true value array for storage (default is 639).\
    """\
\
    # Load averaged hyperparameters\
    avg_results_df = pd.read_csv(os.path.join(data_path, 'avg_results.csv'))\
    cities = avg_results_df['city'].values\
\
    # Initialize arrays for predictions and true values\
    all_true_values = np.full((len(cities), max_length), np.nan)\
    all_predicted_values = np.full((len(cities), max_length), np.nan)\
    all_probabilities = np.full((len(cities), max_length), np.nan)\
    all_true_values2009 = np.full((len(cities), max_length), np.nan)\
    le = LabelEncoder()\
\
    for i, city in enumerate(cities):\
        # Retrieve averaged parameters for the city\
        city_params = avg_results_df[avg_results_df['city'] == city].iloc[0]\
        model_params = \{\
            'n_estimators': int(city_params['n_estimators']),\
            'max_depth': int(city_params['max_depth']),\
            'min_samples_split': int(city_params['min_samples_split']),\
            'min_samples_leaf': int(city_params['min_samples_leaf']),\
            'bootstrap': city_params['bootstrap'],\
            'class_weight': 'balanced',\
            'random_state': 0\
        \}\
\
        # Load city data for years 2009 and 2014\
        file_path_start = os.path.join(data_path, f"\{city.lower()\}_cd_2009_tech.csv")\
        file_path_end = os.path.join(data_path, f"\{city.lower()\}_cd_2014_tech.csv")\
\
        try:\
            # Load data and preprocess\
            df_start = pd.read_csv(file_path_start)\
            df_end = pd.read_csv(file_path_end)\
            df_start.columns = [f'IPC\{i+1\}' for i in range(len(df_start.columns))]\
            df_end.columns = [f'IPC\{i+1\}' for i in range(len(df_end.columns))]\
            df_train = df_start.applymap(lambda x: 1 if x > 0 else 0)\
            df_test = df_end.applymap(lambda x: 1 if x > 0 else 0)\
\
            # Prepare training and test data\
            X = df_start.iloc[1:, :].values.T\
            y = df_test.iloc[0, :].values.flatten()\
            y_2009 = df_train.iloc[0, :].values.flatten()\
            scaler = StandardScaler()\
            X = scaler.fit_transform(X)\
\
            city_true_values = np.full(max_length, np.nan)\
            city_predicted_values = np.full(max_length, np.nan)\
            city_probabilities = np.full(max_length, np.nan)\
\
            # Cross-validation training and prediction\
            segment_length = int(len(y) / 10)\
            for start in range(0, len(y), segment_length):\
                end = min(start + segment_length, len(y))\
                X_train = np.concatenate([X[:start], X[end:]])\
                y_train = np.concatenate([y[:start], y[end:]])\
                X_test = X[start:end]\
                y_test = y[start:end]\
\
                smote = SMOTEN(random_state=7, sampling_strategy='not majority')\
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)\
\
                model = RandomForestClassifier(**model_params)\
                model.fit(X_train_resampled, le.fit_transform(y_train_resampled))\
\
                y_pred = model.predict(X_test)\
                y_prob = model.predict_proba(X_test)[:, 1]\
\
                city_true_values[start:end] = y_test\
                city_predicted_values[start:end] = y_pred\
                city_probabilities[start:end] = y_prob\
\
            # Store predictions\
            all_true_values[i, :len(city_true_values)] = city_true_values\
            all_predicted_values[i, :len(city_predicted_values)] = city_predicted_values\
            all_probabilities[i, :len(city_probabilities)] = city_probabilities\
            all_true_values2009[i, :len(y_2009)] = y_2009\
\
        except FileNotFoundError:\
            print(f"File not found for \{city\}. Skipping...")\
        except Exception as e:\
            print(f"An error occurred for \{city\}: \{e\}")\
\
    # Save results\
    pd.DataFrame(all_true_values, index=cities).T.to_csv(os.path.join(output_path, 'true_rf.csv'), index=False)\
    pd.DataFrame(all_predicted_values, index=cities).T.to_csv(os.path.join(output_path, 'pred_rf.csv'), index=False)\
    pd.DataFrame(all_probabilities, index=cities).T.to_csv(os.path.join(output_path, 'prob_rf.csv'), index=False)\
    pd.DataFrame(all_true_values2009, index=cities).T.to_csv(os.path.join(output_path, 'true_rf_2009.csv'), index=False)\
\
# Example usage\
# evaluate_rf_with_avg_params(data_path='path/to/data_directory', output_path='path/to/output_directory')\
}