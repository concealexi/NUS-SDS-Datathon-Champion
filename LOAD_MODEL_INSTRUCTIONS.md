# Model Loading Instructions
This document provides instructions on how to load and use the trained model for the NUS_DATATHON_CHAMPION_TEAM 256 project.

## Prerequisites
- Python 3.8 or above
- Required Libraries: pandas, numpy, scikit-learn, joblib, tensorflow, xgboost, seaborn, statsmodels, matplotlib
  Install them using the command: pip install pandas numpy scikit-learn
  
## Model Loading Instructions

1. Ensure all prerequisites are installed.

2. Download the model file from the main repositry NUS_DATATHON_CHAMPION_TEAM 256.

3. Load the model

## Usage Example

Once the model is loaded, you can use it to make predictions as follows:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # Replace with the actual model class you used
import joblib
from sklearn.preprocessing import LabelEncoder

def testing_hidden_data(hidden_data: pd.DataFrame) -> list:
    '''DO NOT REMOVE THIS FUNCTION.

    The function accepts a dataframe as input and returns an iterable (list)
    of binary classes as output.

    The function should be coded to test on hidden data
    and should include any preprocessing functions needed for your model to perform.

    All relevant code MUST be included in this function.
    '''

    # Load the trained model
    model = joblib.load('./base_model.joblib')

    # Assume that you have a preprocessing function
    def preprocess_data(data):
        df2 = df2.dropna(subset=["LATITUDE", "LONGITUDE"])
        df2 = df2[df2['Company Status (Active/Inactive)'] == 'Active']
        # to ensure important info isnt being lost
        df2['Import/Export Status_Missing'] = df2['Import/Export Status'].isna()

        df3 = df2.dropna(subset=["Employees (Single Site)", "Employees (Domestic Ultimate Total)", "Employees (Global Ultimate Total)",
                                       "Year Found"])

        df3 = pd.get_dummies(df2, columns=['Entity Type'], prefix='Entity_Type')
        df3 = pd.get_dummies(df2, columns=['Ownership Type'], prefix='Ownership_Type')
        df3 = pd.get_dummies(df2, columns=['Import/Export Status'], prefix='Import_Export_Status')
        df3 = df2.drop(columns=[col for col in ["Company Status (Active/Inactive)", "Entity Type", 'Import/Export Status', 'Ownership Type'] if col in df2.columns], errors='ignore') # are all Active

        # Find the frequency of each industry
        sic_code_frequency = df3['SIC Code'].value_counts()

        # Set a threshold for low-frequency SIC Codes
        threshold = 7  # Adjust this threshold based on your preference

        # Identify SIC Codes with frequency below the threshold
        low_frequency_sic_codes = sic_code_frequency[sic_code_frequency < threshold].index

        # Replace these low-frequency SIC Codes with a common label "Others"
        df3['SIC Code'] = df3['SIC Code'].replace(low_frequency_sic_codes, 'Others')

        sic_code_frequency1 = df3['SIC Code'].value_counts()
        df3['SIC Code'] = df3['SIC Code'].astype(str)
        # Use label encoding for 'SIC Code'
        label_encoder = LabelEncoder()
        df3['SIC Code'] = label_encoder.fit_transform(df3['SIC Code'])
        # Specify columns to drop
        columns_to_drop = ["error", "Fiscal Year End", "Global Ultimate Company", "Domestic Ultimate Company", "Web Address", "Square Footage", "Company Description", "PostCode", "8-Digit SIC Code", "8-Digit SIC Description", "AccountID",
                   "Parent Company", "City", "Country", "Address", "Address1", "Industry", "Region", "Parent Country", "Global Ultimate Country", "Company"]

        # Drop columns if they exist in the DataFrame
        df4 = df3.drop(columns=[col for col in columns_to_drop if col in df3.columns], errors='ignore')
        df4 = df4.dropna(subset=["Employees (Single Site)", "Employees (Domestic Ultimate Total)", "Employees (Global Ultimate Total)",
                                       "Year Found"])
        scaler = StandardScaler()
        preprocessed_data = scaler.fit_transform(data)
        return preprocessed_data

    # Preprocess the hidden data
    preprocessed_hidden_data = preprocess_data(hidden_data)

    # Make predictions using the trained model
    predictions = model.predict(preprocessed_hidden_data)

    return predictions.tolist()
