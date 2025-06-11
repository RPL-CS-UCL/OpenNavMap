#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="DP search over a difference matrix with velocity and re-localization."
    )
    parser.add_argument('--npy_file', type=str, required=True,
                        help="Path to .npy file containing D_all of shape (n_query, n_db).")
    parser.add_argument('--vMin', type=float, default=0.6,
                        help="Minimum velocity in index units per step.")
    parser.add_argument('--vMax', type=float, default=2.4,
                        help="Maximum velocity in index units per step.")
    parser.add_argument('--numVel', type=int, default=20,
                        help="Number of velocities to sample between vMin and vMax.")
    parser.add_argument('--topK', type=int, default=5,
                        help="Number of top start indices (smallest D_all[0]) to try.")
    parser.add_argument('--seqlen', type=int, default=10,
                        help="Step size for sampling relocation region.")
    return parser.parse_args()

def dp_search_paths_with_backtrack(D_all, vMin, vMax, numVel, topK, seqLen=3.0):
    """
    DP with storing layer states for backtracking.
    Returns all end states with weights, best path list of (i, j).
    """
    eps = 1e-8
    mu = 1
    cost_penalty = 0.3
    
    n_query, n_db = D_all.shape
    cost_matrix = np.power(D_all, mu)

    column_indices = np.arange(0, n_query).tolist()
    velocities = np.linspace(vMin, vMax, numVel).tolist()
    print(f"Velocities: {velocities}")

    # layer_states[j]: dict mapping (i, v) → (best_weight, prev_i, prev_v)
    layer_states = []

    # Initialize layer 0
    state0 = {}
    for i_coord in column_indices:
        cost = cost_matrix[i_coord, 0]
        state0[i_coord] = (cost, i_coord, None, None)
    layer_states.append(state0)

    # Iterate layers
    for j in range(0, n_db - 1):
        curr_states = layer_states[j]
        next_states = {}

        for i_coord, (cum_cost, i_cum, _, prev_v) in curr_states.items():
            if prev_v is None:
                vel_list = velocities
            else:
                vel_list = [prev_v]

            # Local search by moving one step at a time
            for v in vel_list:
                i_next = i_cum + v
                i_coord_next = int(i_next)
                if 0 <= i_coord_next < n_query:
                    cost1 = cost_matrix[i_coord_next, j+1]
                    new_cost = cum_cost + cost1
                    key = i_coord_next
                    prev_best = next_states.get(key, (np.inf, None, None, None))[0]
                    if new_cost < prev_best:
                        next_states[key] = (new_cost, i_next, i_coord, v) # keep the same velocity

            # Relocation sampling by jumping to another sequence
            lower_i, upper_i = i_coord - seqLen, i_coord + seqLen
            for k in range(0, int(lower_i)):
                cost2 = cost_matrix[k, j+1]
                new_cost2 = cum_cost + cost2 + cost_penalty # regularization for not using sequence
                key2 = k
                prev_best2 = next_states.get(key2, (np.inf, None, None, None))[0]
                if new_cost2 < prev_best2:
                    next_states[key2] = (new_cost2, k, i_coord, v) # keep the same velocity

            for k in range(int(upper_i), n_query):
                cost2 = cost_matrix[k, j+1]
                new_cost2 = cum_cost + cost2 + cost_penalty # regularization for not using sequence
                key2 = k
                prev_best2 = next_states.get(key2, (np.inf, None, None, None))[0]
                if new_cost2 < prev_best2:
                    next_states[key2] = (new_cost2, k, i_coord, v) # keep the same velocity

        if not next_states:
            print(f"No surviving states at layer {j+1}, stopping early.")
            layer_states.append({})
            break

        layer_states.append(next_states)

    # Final layer index
    final_j = len(layer_states) - 1
    final_states = layer_states[final_j]
    if not final_states:
        print("No complete path.")
        return [], None

    # Find best end state
    best_weight, best_state = np.inf, None
    for i_final, (cum_cost, _, _, _) in final_states.items():
        if cum_cost < best_weight:
            best_weight = cum_cost
            best_state = i_final

    # Backtrack
    path = []
    curr = best_state  # i at layer final_j
    for j in range(final_j, -1, -1):
        i_curr = curr
        path.append((i_curr, j))
        # get predecessor
        cum_cost, _, prev_i, prev_v = layer_states[j][i_curr]
        curr = prev_i  # for next iteration
    path.reverse()

    return cost_matrix, list(final_states.items()), path  # return all end-state info and best path
def plot_results(D_all, cost_matrix, best_path, fig_path):
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111)
    im = ax.imshow(cost_matrix, cmap='Greys', aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    im.set_clim(0.0, 1.0)
    ax.set_xlabel('Query Index', fontsize=14)
    ax.set_ylabel('Database Index', fontsize=14)
    ax.set_title("Cost Matrix", fontsize=16)

    if best_path:
        ds = [i for i, j in best_path]
        qs = [j for i, j in best_path]
        ax.plot(qs, ds, 'b.', markersize=8, alpha=1.0, label='Best Path', markeredgewidth=1)
        ax.plot(qs, ds, 'b-', linewidth=1, alpha=1.0)
        ax.legend()

    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(fig_path)

def main():
    args = parse_args()

    D_all = np.load(args.npy_file)
    if D_all.ndim != 2:
        raise ValueError("Loaded array must be 2D (n_db, n_query).")

    # Run DP search
    cost_matrix, end_states, best_path = dp_search_paths_with_backtrack(
        D_all, args.vMin, args.vMax, args.numVel, args.topK)

    plot_results(D_all, cost_matrix, best_path, args.npy_file.replace('.npy', '_search.jpg'))

if __name__ == "__main__":
    main()
