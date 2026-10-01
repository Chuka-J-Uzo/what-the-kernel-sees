import re
import csv

# Define the input and output file paths
input_file = './../output-Phi/Phi-perf-heaptrack-wasmedge-stats.txt'
output_file = 'output.csv'

# Define the regex pattern to extract parameters and their corresponding values
pattern = r'-\s+([^\s=]+)\s*=\s*(.*)$'

# Define the patterns to filter out
patterns_to_filter = [
    'ggml_init_cublas',
    'llama_model_loader',
    'llm_load_vocab',
    'llm_load_tensors',
    'llama_new_context_with_model',
    'llama_kv_cache_init',
    'llama_print_timings',
    'Performance counter stats for',
    '\.{10,}'
]

# Open the input file and read its contents
with open(input_file, 'r') as file:
    data = file.readlines()

# Filter out unwanted patterns and extract parameter-value pairs
param_values = []
for line in data:
    if not any(pattern in line for pattern in patterns_to_filter):
        match = re.search(pattern, line)
        if match:
            parameter = match.group(1).strip()
            value = match.group(2).strip()
            param_values.append((parameter, value))

# Write the extracted parameter-value pairs to a CSV file
with open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Parameter', 'Value'])  # Write header row
    writer.writerows(param_values)

print("Extraction and CSV creation completed successfully.")
