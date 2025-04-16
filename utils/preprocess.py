import os
import pandas as pd

def process_file(file_path):
    """Process an individual CSV file."""
    try:
        df = pd.read_csv(file_path)

        # Example processing: remove rows with missing values
        df.dropna(inplace=True)

        # You can customize more processing here
        # Example: df = df[df["column_name"] > 0]

        return df

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_all_patients(input_folder, output_folder):
    """Process all CSV files in the input folder and save to output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    for file in files:
        input_path = os.path.join(input_folder, file)
        print(f"Processing {file}...")

        df = process_file(input_path)

        if df is not None:
            output_path = os.path.join(output_folder, f"processed_{file}")
            df.to_csv(output_path, index=False)
            print(f"Saved processed file to: {output_path}")
        else:
            print(f"Skipped {file} due to processing error.")
