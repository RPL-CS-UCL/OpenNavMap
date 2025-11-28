#!/usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

from python.utils.utils_setting_color_font import *

setting_font(fontsize=22, titlesize=22, legend_fontsize=22)

def plot_results(args, df, methods, colors, markers, linestyles):
    fig = plt.figure(figsize=(12, 6.0))
    gs = plt.GridSpec(1, 2, width_ratios=[1.8, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    plt.subplots_adjust(wspace=-1.5)

    if 'reloc3r' in methods:
        filter_methods = [method for method in methods if 'vpr_netvlad' not in method]
    else:
        filter_methods = methods
    unique_top_k = df[df['Method'] == filter_methods[0]]['TOP_K'].unique()
    for idx, method in enumerate(filter_methods):
        method_data = df[df['Method'] == method]
        label = method_data['Abbreviation'].iloc[0]
        
        ax1.plot(
            method_data['TOP_K'].values,
            method_data['Precision @ Pose Error < (100.0cm, 10deg)'].values,
            marker=markers[int(idx/2)],
            markersize=9,
            color=colors[int(idx/2)],
            linestyle=linestyles[idx % 2],
            label=label,
            linewidth=2.0
        )

    ax1.set_xlabel('Number of Reference Images')
    ax1.set_ylabel('Precision@[100cm, 10°] (\%)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(ncol=1, fontsize=17.5, loc='lower left')
    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    ax1.set_xticks(unique_top_k)

    if 'reloc3r' in methods:
        filter_methods = [method for method in methods if 'vpr_netvlad' not in method]
    else:
        filter_methods = methods
    unique_top_k = []
    for idx, method in enumerate(filter_methods):
        method_data = df[df['Method'] == method]
        label = method_data['Abbreviation'].iloc[0]
        if 'duster' in method or 'master' in method:
            unique_top_k = method_data['TOP_K'].unique()
            ax2.plot(
                method_data['TOP_K'], 
                method_data['AUC @ Pose Error < (100.0cm, 10deg)'],
                marker=markers[int(idx/2)], 
                markersize=9,
                color=colors[int(idx/2)],
                linestyle=linestyles[idx % 2],
                label=label,
                linewidth=2.0
            )
    
    ax2.set_xlabel('Number of Reference Images')
    ax2.set_ylabel('AUC@[100cm, 10°] (\%)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(ncol=1, fontsize=17.5, loc='upper left')
    ax2.set_xticks(unique_top_k)
    ax2.tick_params(axis='x', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(args.csv_path), 'rpe_results.pdf')
    plt.savefig(output_path, dpi=300)
    output_path = os.path.join(os.path.dirname(args.csv_path), 'rpe_results.png')
    plt.savefig(output_path, dpi=300)
    plt.close()    

def generate_latex_table(args, df, methods, colors, markers, linestyles):
    """
    Generate LaTeX table with format "Precision Value/AUC Value"
    """
    filter_methods = methods

    unique_top_k = sorted(df[df['Method'] == filter_methods[0]]['TOP_K'].unique())
    
    latex_table = []
    latex_table.append("\\begin{table}[htbp]")
    latex_table.append("\\centering")
    latex_table.append("\\begin{tabular}{l" + "c" * len(filter_methods) + "}")
    latex_table.append("\\hline")
    
    header = "\\textbf{Top-K} "
    for method in filter_methods:
        method_data = df[df['Method'] == method]
        abbreviation = method_data['Abbreviation'].iloc[0]
        header += f"& \\textbf{{{abbreviation}}} "
    header += "\\\\"
    latex_table.append(header)
    latex_table.append("\\hline")
    
    for top_k in unique_top_k:
        row = f"{top_k} "
        for method in filter_methods:
            method_data = df[(df['Method'] == method) & (df['TOP_K'] == top_k)]
            if not method_data.empty:
                precision = method_data['Precision @ Pose Error < (100.0cm, 10deg)'].iloc[0]
                auc = method_data['AUC @ Pose Error < (100.0cm, 10deg)'].iloc[0]
                row += f"& ${precision:.1f}/{auc:.1f}$ "
            else:
                row += "& - "
        row += "\\\\"
        latex_table.append(row)
    
    latex_table.append("\\hline")
    latex_table.append("\\end{tabular}")
    latex_table.append("\\caption{Precision/AUC values for different methods and Top-K settings.}")
    latex_table.append("\\label{tab:rpe_results}")
    latex_table.append("\\end{table}")
    
    output_path = os.path.join(os.path.dirname(args.csv_path), 'rpe_results_table.tex')
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_table))
    
    print(f"LaTeX table saved to: {output_path}")
    print("\nLaTeX Table:")
    print('\n'.join(latex_table))

def main():
    parser = argparse.ArgumentParser(description='Visualize RPE results')
    parser.add_argument('--csv_path', type=str, required=True,
                      help='Path to the CSV file containing RPE results')
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    methods = df['Method'].unique()

    colors = acquire_color_palette()
    markers = acquire_marker()
    linestyles = acquire_linestyle()

    plot_results(args, df, methods, colors, markers, linestyles)
    generate_latex_table(args, df, methods, colors, markers, linestyles)

if __name__ == '__main__':
    main()
