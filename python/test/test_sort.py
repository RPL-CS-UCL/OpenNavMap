import numpy as np

all_id = np.array([0, 1, 2, 3, 4])
all_dis = np.array([0.1, 0.3, 0.2, 0.5, 0.4])
sorted_indices = np.argsort(all_dis)
sorted_dis = all_dis[sorted_indices]
sorted_ids = all_id[sorted_indices]
print(sorted_dis)
print(sorted_ids)