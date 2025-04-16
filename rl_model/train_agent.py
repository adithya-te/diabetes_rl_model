import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Explicit folder paths for preprocessed and raw data
PREPROCESSED_FOLDER = "C:/Users/adiad/OneDrive/Desktop/diabetes_rl_project/hupa_ucm_diabetes_dataset/Preprocessed"
RAW_DATA_FOLDER = "C:/Users/adiad/OneDrive/Desktop/diabetes_rl_project/hupa_ucm_diabetes_dataset/Raw_Data"

def load_preprocessed_data(preprocessed_folder):
    try:
        # List all .csv files in the preprocessed data directory
        patient_files = [f for f in os.listdir(preprocessed_folder) if f.endswith('.csv')]
        print(f"Patient files found: {patient_files}")  # Print the list of files

        patient_data = []
        for file in patient_files:
            file_path = os.path.join(preprocessed_folder, file)
            print(f"Checking file: {file_path}")
            
            if os.path.isfile(file_path):
                # Load the .csv file
                patient_df = pd.read_csv(file_path, sep=';')
                print(f"Loaded data for {file}")
                patient_data.append(patient_df)
            else:
                print(f"File {file_path} not found!")
        
        if len(patient_data) == 0:
            print("No preprocessed data found!")
        return patient_data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

def preprocess_data(patient_data):
    try:
        # Combine all the patient data into a single dataframe
        combined_df = pd.concat(patient_data, ignore_index=True)
        
        print(f"Columns in loaded DataFrame: {combined_df.columns}")
        print(f"Preview:\n{combined_df.head()}")

        # Drop non-numeric columns (e.g., 'time' column) to focus only on numeric columns
        numeric_df = combined_df.select_dtypes(include=[np.number])

        # Preprocessing - Handle missing values (drop rows with NaN)
        numeric_df = numeric_df.dropna()

        # If there's no numeric data after conversion, raise an error
        if numeric_df.empty:
            raise ValueError("No numeric data available after conversion.")

        return numeric_df
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

def scale_data(df):
    try:
        # Scaling the numeric data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        return scaled_data
    except Exception as e:
        print(f"Error during scaling: {e}")
        return None

def build_model(input_shape):
    try:
        # Define a simple neural network model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    except Exception as e:
        print(f"Error during model building: {e}")
        return None

def train_agent():
    print("Checking folder paths...")
    # Check if the directories exist
    preprocessed_folder_exists = os.path.exists(PREPROCESSED_FOLDER)
    raw_data_folder_exists = os.path.exists(RAW_DATA_FOLDER)

    print(f"Preprocessed folder exists: {preprocessed_folder_exists}")
    print(f"Raw data folder exists: {raw_data_folder_exists}")

    if not preprocessed_folder_exists or not raw_data_folder_exists:
        print("Error: Required folder(s) not found!")
        return

    # Load preprocessed data
    print("Loading preprocessed patient data...")
    patient_data = load_preprocessed_data(PREPROCESSED_FOLDER)

    if not patient_data:
        print("No data found for patients.")
        return

    # Preprocess the data
    preprocessed_df = preprocess_data(patient_data)

    if preprocessed_df is None:
        print("Preprocessing failed. Exiting.")
        return

    # Scale the data
    scaled_data = scale_data(preprocessed_df)

    if scaled_data is None:
        print("Data scaling failed. Exiting.")
        return

    print(f"Data scaling successful, shape of scaled data: {scaled_data.shape}")

    # Build and train the model
    model = build_model((scaled_data.shape[1],))

    if model is None:
        print("Model building failed. Exiting.")
        return

    # For the sake of example, let's just train the model with the scaled data
    # We need target values for the training, which we can get from the 'glucose' column, for example
    target = preprocessed_df['glucose']

    # Train the model (using a dummy dataset here for illustration)
    model.fit(scaled_data, target, epochs=10, batch_size=32)

    print("Model training complete!")

if __name__ == "__main__":
    train_agent()
