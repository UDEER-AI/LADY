import os
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from multiprocessing import Pool, cpu_count

REPO_ROOT = Path(os.environ.get("NAVSIM_DEVKIT_ROOT", Path(__file__).resolve().parents[1]))
EXP_ROOT = Path(os.environ.get("NAVSIM_EXP_ROOT", REPO_ROOT / "exp"))
VIS_ROOT = Path(os.environ.get("NAVSIM_VIS_ROOT", EXP_ROOT / "vis"))
TRAJ_DUMP_PATH = Path(os.environ.get("TRAJ_DUMP_PATH", VIS_ROOT / "traj_dump.pkl"))

def kmeans_trajs(trajs: dict[int,[np.ndarray]], K: int):
    all_trajs = []
    # get values of trajs
    for _,traj in trajs.items():
        all_trajs.extend(traj)

    traj_arr = np.stack(all_trajs)  # shape: (N, T, 3)
    traj_arr = traj_arr[:,:,:2]

    traj_flat = traj_arr.reshape(len(traj_arr), -1)  # shape: (N, T*2)

    kmeans = KMeans(n_clusters=K).fit(traj_flat)
    centers = kmeans.cluster_centers_.reshape(K, -1, 2)  # shape: (k, T, 2)
    for i in range(K):
        center = centers[i]
        plt.plot(center[:, 0], center[:, 1], label=f"Cluster {i}", linewidth=3)
    output_dir = VIS_ROOT / "kmeans"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'plan_{K}.png', bbox_inches='tight')
    plt.close()
    np.save(output_dir / f'plan_vocab_{K}_kmeans.npy', centers)


def kmeans_trajs(trajs: dict[int,[np.ndarray]], K: int):
    cluster_dir = VIS_ROOT / "cluster" / "kmeans"
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
    all_trajs = []
    # get values of trajs
    for _,traj in trajs.items():
        all_trajs.extend(traj)

    traj_arr = np.stack(all_trajs)  # shape: (N, T, 3)
    traj_arr = traj_arr[:,:,:2]

    traj_flat = traj_arr.reshape(len(traj_arr), -1)  # shape: (N, T*2)

    kmeans = KMeans(n_clusters=K).fit(traj_flat)
    centers = kmeans.cluster_centers_.reshape(K, -1, 2)  # shape: (k, T, 2)
    for i in range(K):
        center = centers[i]
        plt.plot(center[:, 0], center[:, 1], label=f"Cluster {i}", linewidth=3)
    fig_path = os.path.join(cluster_dir, f'plan_vocab_{K}_kmeans.png')
    npy_path = os.path.join(cluster_dir, f'plan_vocab_{K}_kmeans.npy')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    np.save(npy_path, centers)

def run_kmeans(k):
    return kmeans_trajs(traj, k)

if __name__ == "__main__":
    with open(TRAJ_DUMP_PATH, "rb") as f:
        traj = pickle.load(f)
    # for k in tqdm(range(20,2000,20)):
    #     kmeans_trajs(traj,k)
    ks = list(range(20, 2000, 20))
    with Pool(processes=8) as pool:
        list(tqdm(pool.imap_unordered(run_kmeans, ks), total=len(ks)))
