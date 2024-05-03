import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import streamlit as st

# Load the dataset
cancer_df = pd.read_csv('cervical_cancer.csv')

# Preprocess the data (replace '?' with NaN, convert to numeric, and fill NaN with mean)
cancer_df = cancer_df.replace('?', np.nan)
cancer_df = cancer_df.drop(columns = ['STDs: Time since first diagnosis' , 'STDs: Time since last diagnosis'])
cancer_df = cancer_df.apply(pd.to_numeric)
cancer_df = cancer_df.fillna(cancer_df.mean())

# Create the input and target dataframes
target_column_name = 'Biopsy'  # Specify the target column name
target_df = cancer_df[target_column_name]
input_df = cancer_df.drop(columns=[target_column_name])

# Create the scaler object and fit it to the input data
scaler = StandardScaler()
X = np.array(input_df).astype('float32')
scaler.fit(X)

# Load the saved model
model = joblib.load('xgb_model.pkl')

# Function to make predictions
def predict(data):
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return prediction[0]

# Streamlit app
st.title("Cervical Cancer Prediction")

# Get user input
age = st.number_input("Age", min_value=0, value=30)
num_partners = st.number_input("Number of sexual partners", min_value=0, value=2)
first_intercourse = st.number_input("Age at first sexual intercourse", min_value=0, value=18)
num_pregnancies = st.number_input("Number of pregnancies", min_value=0, value=1)
smokes = st.selectbox("Smokes", [0, 1], index=0)
smokes_years = st.number_input("Years of smoking", min_value=0, value=0)
smokes_packs = st.number_input("Packs per year", min_value=0, value=0)
hormonal_contraceptives = st.selectbox("Hormonal Contraceptives", [0, 1], index=1)
hormonal_years = st.number_input("Years of Hormonal Contraceptives", min_value=0, value=5)
iud = st.selectbox("IUD", [0, 1], index=0)
iud_years = st.number_input("Years of IUD", min_value=0, value=0)

# Create a new data sample with all features
new_data = pd.DataFrame({
    'Age': [age],
    'Number of sexual partners': [num_partners],
    'First sexual intercourse': [first_intercourse],
    'Num of pregnancies': [num_pregnancies],
    'Smokes': [smokes],
    'Smokes (years)': [smokes_years],
    'Smokes (packs/year)': [smokes_packs],
    'Hormonal Contraceptives': [hormonal_contraceptives],
    'Hormonal Contraceptives (years)': [hormonal_years],
    'IUD': [iud],
    'IUD (years)': [iud_years]
})

# Fill missing columns with the mean of the original dataset
missing_cols = set(input_df.columns) - set(new_data.columns)
for col in missing_cols:
    new_data[col] = input_df[col].mean()

# Make sure the columns are in the same order as the input dataset
new_data = new_data[input_df.columns]

# Make prediction
if st.button("Predict"):
    prediction = predict(new_data)
    if prediction == 1:
        st.write("The patient is likely to have cervical cancer.")
    else:
        st.write("The patient is not likely to have cervical cancer.")
