#!/usr/bin/env python

import json
import pandas as pd
import argparse

def parse_report_eval(path_report):
    # Initialize an empty dictionary to hold the data
    data = {}

    # Open the file and read the lines
    with open(path_report, 'r') as file:
        lines = file.readlines()

    # Initialize an empty string to hold the current method's data
    current_data = ''

    # Iterate over the lines in the file
    for line in lines:
        # If the line starts with 'Evaluate', it's a new method
        if line.startswith('Evaluate'):
            # If there's data in current_data, add it to the main data dictionary
            if current_data:
                data[current_method] = json.loads(current_data)
            # Get the new method name
            current_method = line.split(': ')[1].strip()
            # Reset current_data
            current_data = ''
        # If the line is a JSON line, add it to current_data
        elif line.strip().startswith('{') or line.strip().startswith('}') or line.strip().startswith('"'):
            current_data += line.strip()

    # Add the last method's data to the main data dictionary
    if current_data:
        data[current_method] = json.loads(current_data)

    for key, values in data.items():
        for key2, values2 in values.items():
            if 'Precision' in key2 or 'AUC' in key2 or 'Estimates' in key2:
                values[key2] *= 100
            if values[key2] < 0.1:
                values[key2] = '{:.3f}'.format(values[key2])
            else:
                values[key2] = '{:.2f}'.format(values[key2])

    df = pd.DataFrame(data).T
    return df

def parse_report_runtime(path_report):
    # Initialize an empty dictionary to hold the data
    data = {}

    # Open the file and read the lines
    with open(path_report, 'r') as file:
        lines = file.readlines()

    # Iterate over the lines in the file
    for line in lines:
        # Split the line into method and runtime
        method, runtime = line.split(': ')
        # Remove the prefix from the method name
        method = method.replace('(image_matching + pose_solver) ', '')
        # Add the data to the dictionary
        data[method] = float(runtime.strip()[:-1])  # Remove the 's' from the end and convert to float

    for key in data.keys():
        data[key] *= 1000
        data[key] = '{:.0f}'.format(data[key])

    df = pd.DataFrame(list(data.items()), columns=['Method', 'Runtime [ms]'])
    return df

def main():
    parser = argparse.ArgumentParser(description='Parse report table from a text file.')
    parser.add_argument('--path_report_eval', type=str, default=None, help='Path to the report of evaluation')
    parser.add_argument('--path_report_runtime', type=str, default=None, help='Path to the report of runtimg')
    args = parser.parse_args()

    if args.path_report_eval is not None:
        df = parse_report_eval(args.path_report_eval)
        print(df)
        path_report_eval_csv = args.path_report_eval.replace('.txt', '.csv')
        df.to_csv(path_report_eval_csv)

    if args.path_report_runtime is not None:
        df = parse_report_runtime(args.path_report_runtime)
        print(df)
        path_report_eval_csv = args.path_report_runtime.replace('.txt', '.csv')
        df.to_csv(path_report_eval_csv)        

if __name__ == '__main__':
    main()
