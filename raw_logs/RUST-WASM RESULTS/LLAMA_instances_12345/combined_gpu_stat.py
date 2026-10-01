import pandas as pd
import os
from typing import Union, Tuple

InputCSVFile = str
ColumnMapping = Tuple[str, str]
CleanedDF = pd.DataFrame

ALLOWED_UNITS = {'%': lambda x: x / 100, 'MiB': lambda x: x / (1024**2), 'GB': lambda x: x * 1024**3, 'W': lambda x: x}

def process_csv(csv_file: InputCSVFile, column_mapping: ColumnMapping) -> CleanedDF:
    """Reads a CSV file, processes it, and returns a cleaned DataFrame."""
    
    def handle_units(col_name: str, value: str) -> Union[float, None]:
        """Handles units and converts to a scalar."""
        if value == "-":
            return None
        elif col_name in ['date_time']:
            return value
        else:
            split_value = value.split()
            if len(split_value) >= 2:
                potential_unit = split_value[-1]
                if potential_unit in ALLOWED_UNITS:
                    scaled_value = ALLOWED_UNITS[potential_unit](float(split_value[0]))
                    return scaled_value
    def apply_handling(row):
        nonlocal column_mapping
        result = {}
        for column, _ in column_mapping:
            bool_mask = row.str.lower().duplicated(keep=False) & row.str.lower() == column
            values = row[bool_mask][column].values
            for value in values:
                rescaled_value = handle_units(column, value)
                if rescaled_value is not None:
                    result[column + '_scaled'] = rescaled_value
                    break
        return pd.Series(result)

    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Print the columns of the current CSV file being processed
    print('Columns in current CSV file:', df.columns)

    cleaned_rows = []
    for index, row in df.iterrows():
        cleaned_row = apply_handling(row)
        if not cleaned_row.empty:
            cleaned_rows.append(cleaned_row)

    # Create DataFrame from cleaned rows
    df_cleaned = pd.concat(cleaned_rows, axis=1).transpose().reset_index(drop=True)

    return df_cleaned

def main(input_folder: str, output_file: str, column_mapping: ColumnMapping) -> None:
    # Process each CSV file and clean up the data
    cleaned_dfs = []
    for csv_file in os.listdir(input_folder):
        if csv_file.endswith(".csv"):
            csv_file_path = os.path.join(input_folder, csv_file)
            cleaned_df = process_csv(csv_file_path, column_mapping)
            cleaned_dfs.append(cleaned_df)

    if cleaned_dfs:
        # Combine the cleaned dataframes
        combined_df = pd.concat(cleaned_dfs, ignore_index=True)

        # Save the combined dataframe to a CSV file
        combined_df.to_csv(output_file, index=False)
        print("Combined GPU stats saved to", output_file)
    else:
        print("No CSV files found in the input folder.")

if __name__ == "__main__":
    input_folder = "/home/blackjack/GITHUB PROJECTS/web-dep/RESULTS_BOX/RUST-WASM RESULTS/LLAMA_instances_12345/"
    output_file = "combined_gpu_stat.csv"

    column_mapping = (
        ('date_time', 'Date and Time'),
        ('gpuutilizationpercent', 'GPU Utilization (%)'),
        ('memutilizationpercent', 'Memory Utilization (%)'),
        ('totalgpumemorymib', 'Total GPU Memory (MiB)'),
        ('freegpumemorymib', 'Free GPU Memory (MiB)'),
        ('usedgputemp', '% Used GPU Memory'),
        ('powerdrawwatt', 'Power Draw (W)'),
        ('temperaturec', 'GPU Temperature (C)'),
        ('totalram', 'Total RAM'),
        ('usedram', '% Used RAM'),
        ('freeram', 'Free RAM'),
        ('cachedram', 'Cached RAM'),
        ('bufferedram', 'Buffered RAM')
    )

    main(input_folder, output_file, column_mapping)