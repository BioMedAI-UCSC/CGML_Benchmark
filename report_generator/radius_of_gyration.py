import mdtraj
import numpy as np

def calc_radius_of_gyration(trajs: list[mdtraj.Trajectory]) -> np.ndarray:
    """Finds the radius of gyration for a list of mdtraj.Trajectory and stores into np array"""
    all_rg = []
    for traj in trajs:
        rg = mdtraj.compute_rg(traj)  # shape: (n_frames,)
        all_rg.extend(rg)
    return np.array(all_rg)