import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../python'))

import csv
import argparse
from matplotlib import pyplot as plt
import numpy as np
from utils.utils_setting_color_font import acquire_color_palette, acquire_marker, setting_font

def load_data(file_path):
	data = {}
	current_scene = None
	current_headers = []
	
	with open(file_path, 'r') as f:
		reader = csv.reader(f)
		while True:
			try:
				row = next(reader)
			except StopIteration:
				break
			if not any(row):
				continue
			if row[0] == 'Scene':
				current_scene = row[1].strip()
				try:
					headers_row = next(reader)
				except StopIteration:
					headers_row = []
				current_headers = [header.strip() for header in headers_row]
			elif current_scene is not None:
				if not row[0].strip():
					current_scene = None
					current_headers = []
				else:
					method = row[0].strip()
					metrics = {}
					for i, header in enumerate(current_headers[1:], start=1):
						if i >= len(row):
							value = ''
						else:
							value = row[i].strip()
						try:
							num_val = float(value)
							if num_val.is_integer():
								num_val = int(num_val)
							metrics[header] = num_val
						except ValueError:
							metrics[header] = value if value else None
					if method not in data:
						data[method] = {}
					data[method][current_scene] = metrics
	return data

def draw_vpr_result(data, result_dir):
	label_each_group = [
		'SgM', 'SgM+GV',
		'SeqM', 'SeqM+GV',
		'SeqMAda', 'SeqMAda+GV',
	]
	N = len(label_each_group)
	scenes = ['s00000', 's00001', 's00002', 's00003', 's00004', 's00005']
	
	group_count = len(data) // N
	
	# Visualization setup
	setting_font()  # Assume this configures fonts as previously discussed
	PALLETE = acquire_color_palette()  # Assume returns color list
	MARKERS = acquire_marker()  # Assume returns marker list

	# Report stat
	for scene in scenes:
		example_method = 'netvlad_VGG16_4096_single_match_1_none'
		total_num_pr = data[example_method][scene]['Total Valid Match Number']
		total_num_query = data[example_method][scene]['Total Query Number']
		print(f"Scene {scene}: Query Number: {total_num_query}, Valid PR: {total_num_pr}")

	for scene in scenes:
		plt.figure(figsize=(7, 3.9))
		ax = plt.gca()
		
		# Group data collection
		group_values = [[] for _ in range(group_count)]
		group_names = [""] * group_count
		methods = list(data.keys())
		for group_id in range(group_count):
			group_methods = methods[group_id*N : (group_id+1)*N]
			group_names[group_id] = data[group_methods[0]][scene]['VPR']
			group_values[group_id] = [
				[data[m][scene]['Max Recall'] for m in group_methods],
				[data[m][scene]['Precision']*100 for m in group_methods],
				[data[m][scene]['Recall']*100 for m in group_methods]
			]

		min_y = min([np.min(values) for values in group_values])

		# Visualization parameters
		x_pos = np.arange(group_count)
		jitter = 0.27  # Horizontal spread within groups
		point_size = 60

		# Create individual points with jitter
		for group_id in range(group_count):
			# Add individual points
			x_jitter = x_pos[group_id] + np.linspace(-jitter, jitter, N)
			# Plot value: max recall
			for i, group_value in enumerate(group_values[0][group_id]):
				for j, (x, y) in enumerate(zip(x_jitter, group_value)):
					ax.scatter(
						x, y, 
						s=point_size,
						c=PALLETE[int(j/2)],
						marker=MARKERS[j],
						zorder=i+1,
						edgecolors='k',
						linewidths=1
					)
				# ax.plot(x_jitter, group_value, c='k', linestyle='--', linewidth=1, zorder=0)
					plt.vlines(
						x, 0, y, 
						color=PALLETE[int(j/2)], linestyles='dashed', linewidth=1
					)

		# Formatting
		ax.set_xticks(x_pos)
		ax.set_xticklabels(group_names, rotation=0, ha='center', fontsize=14)
		ax.set_ylabel('Max Recall@100 Precision [\%]', fontsize=14)
		# ax.set_title(f'Scene {scene[1:]}', fontsize=16)
		ax.grid(True, linestyle='--', alpha=0.7)
		
		# Legend for method variants
		# legend_elements = []
		# legend_elements.append(
		# 	plt.Line2D([0], [0], 
		# 			  marker=MARKERS[0], 
		# 			  color='w', 
		# 			  label='Max Recall@100 Precision[\%]',
		# 			  markersize=10,
		# 			  markerfacecolor=PALLETE[0])
		# )
		# legend_elements.append(
		# 	plt.Line2D([0], [0], 
		# 			  marker=MARKERS[1], 
		# 			  color='w', 
		# 			  label='Precision[\%]',
		# 			  markersize=10,
		# 			  markerfacecolor=PALLETE[0])
		# )
		# legend_elements.append(
		# 	plt.Line2D([0], [0], 
		# 			  marker=MARKERS[2], 
		# 			  color='w', 
		# 			  label='Recall[\%]',
		# 			  markersize=10,
		# 			  markerfacecolor=PALLETE[0])
		# )
		legend_elements = [
			plt.Line2D([0], [0], 
					  marker=MARKERS[i], 
					  color='w', 
					  label=label_each_group[i],
					  markersize=10,
					  markerfacecolor=PALLETE[int(i/2)],
					  markeredgecolor='k')
			for i in range(len(label_each_group))
		]
		ax.legend(handles=legend_elements, 
				 loc='upper center',
				 ncol=3,
				 fontsize=12, bbox_to_anchor=(0.5, 1.02))
		
		plt.ylim([max(min_y - 5.0, -0.5), 100.0])
		plt.tight_layout()

		# Save figure
		os.makedirs(result_dir, exist_ok=True)
		plt.savefig(os.path.join(result_dir, f"maxrecall_{scene}.pdf"), 
				   bbox_inches='tight', 
				   dpi=300)
		plt.close()
		os.system(f'pdfcrop {os.path.join(result_dir, f"maxrecall_{scene}.pdf")}')
		# input()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Load and process CSV data with scene-based metrics')
	parser.add_argument('--csvfile', type=str, help='Path to the CSV file to process')
	parser.add_argument('--result_dir', type=str, help='Path to the result path')
	args = parser.parse_args()

	data = load_data(args.csvfile)

	# Visualization
	draw_vpr_result(data, args.result_dir)

