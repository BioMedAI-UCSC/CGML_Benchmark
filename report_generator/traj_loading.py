import h5py #type: ignore
import os
import mdtraj #type: ignore
import mdtraj.core.topology #type: ignore
from mdtraj.formats.hdf5 import Frames
from numpy.typing import NDArray
import numpy
from dataclasses import dataclass
import preprocess
import pickle
import logging
from pathlib import Path
from report_generator.cache_loading import load_cache_or_make_new
from concurrent.futures import ProcessPoolExecutor, as_completed 
import multiprocessing
from tqdm import tqdm
from functools import partial


def load_path_for_prior(path, prior_name, stride):
    prior_builder = preprocess.prior_types[prior_name]()
    return load_native_path(path, prior_builder, stride)


@dataclass
class ModelTraj:
    trajectory: mdtraj.Trajectory

    def filterFrames(self, start, end):
        return ModelTraj(self.trajectory[start:end])

@dataclass
class NativeTraj:
    trajectory: mdtraj.Trajectory
    forces: NDArray
    path: str

    def stride(self, stride: int):
        return NativeTraj(self.trajectory[::stride], self.forces[::stride], self.path)

@dataclass
class NativeTrajPathNumpy:
    numpy_file: str
    pdb_top_path: str

@dataclass
class NativeTrajPathH5:
    h5_path: str
    pdb_top_path: str

NativeTrajPath = NativeTrajPathH5 | NativeTrajPathNumpy

# this is where AA data re-generated from Cecilia is projected to CG space
def apply_cg(traj, atom_indicies):
    new_traj = traj.atom_slice(atom_indicies)
    assert new_traj.n_atoms == len(atom_indicies)
    return new_traj

def load_model_traj(sim_path: Path) -> ModelTraj:
    traj = mdtraj.load_hdf5(sim_path)
    return ModelTraj(traj)

def load_model_traj_pickle(sim_path: Path) -> list[ModelTraj]:
    with open(sim_path, 'rb') as f:
        struct = pickle.load(f)
        mdtraj_list = struct['mdtraj_list']

    for t in mdtraj_list:
        assert isinstance(t, mdtraj.Trajectory)
    return [ModelTraj(t) for t in mdtraj_list]

def native_traj_iter_loader(
        native_paths: list[NativeTrajPath],
        prior_params,
        stride: int):
    """
    loading all the native paths into memory can cause OOM when also generating a big TICA model, so this just makes an iterator over them so only a few trajs need to be in memory at a time.
    """
    prior_name = prior_params["prior_configuration_name"]
    prior_builder = preprocess.prior_types[prior_name]()

    num_residues = None
    for path in native_paths:
        native_traj = load_native_path(path, prior_builder, stride)
        if num_residues is None:
            num_residues = native_traj.trajectory.n_residues
        assert num_residues == native_traj.trajectory.n_residues, f"BAD DATA, NATIVE PATH HAS THE WRONG NUMBER OF RESIDUES: {native_traj.trajectory.n_residues} vs {num_residues}, at {native_traj.path}"
        yield native_traj
    

def load_native_trajs_stride(
        native_paths: list[NativeTrajPath],
        prior_params,
        stride: int,
        cache_path: Path,
        protein_name: str,
        force_cache_regen: bool,
        temperature) -> tuple[list[NativeTraj], Path]:

    all_native_file_strided = os.path.join(
        cache_path,
        f"{protein_name}_native_trajs_CG_stride{stride}_{temperature}K.pkl"
    )

    def load_no_cache_parallel(native_paths, prior_params, stride):
        prior_name = prior_params["prior_configuration_name"]

        native_trajs = []
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(load_path_for_prior, path, prior_name, stride): path
                for path in native_paths
            }
            for future in tqdm(as_completed(futures), total=len(native_paths), desc="Loading trajectories"):
                native_trajs.append(future.result())

        return native_trajs

    def load_no_cache():
        prior_name = prior_params["prior_configuration_name"]
        prior_builder = preprocess.prior_types[prior_name]()
        native_trajs = [load_native_path(path, prior_builder, stride) for path in native_paths]
        return native_trajs

    native_trajs = load_cache_or_make_new(
        Path(all_native_file_strided),
        partial(load_no_cache_parallel, native_paths, prior_params, stride),
        list,
        force_cache_regen
    )

    return native_trajs, Path(all_native_file_strided)

def load_numpy(numpy_file: str, pdb_file: str) -> NativeTraj:
    trajectory = mdtraj.load_pdb(pdb_file)
    trajectory.xyz = numpy.load(numpy_file) / 10.0 #divide by 10 to convert angstroms to nanometers to match mdtraj
    trajectory.time = numpy.arange(len(trajectory.xyz)) #mdtraj will crash if you slice a trajectory without doing this since the time is still a 1x1 numpy array from loading just from the pdb
    logging.info("DID TIME THING WHEN LOADING")

    return NativeTraj(trajectory, numpy.array([]), "")

def load_native_path(native_path: NativeTrajPath, prior_builder, stride) -> NativeTraj:
    match native_path:
        case NativeTrajPathH5(h5_path, _):
            logging.info(f"loading path: {h5_path}")
        case NativeTrajPathNumpy(numpy_file, pdb_file):
            logging.info(f"loading path: {numpy_file}")
            return load_numpy(numpy_file, pdb_file)

    topology = mdtraj.load(native_path.pdb_top_path).top
    atoms_idx = prior_builder.select_atoms(topology)

    input_traj = mdtraj.load_hdf5(native_path.h5_path)
    out_traj = apply_cg(input_traj, atoms_idx)

    with h5py.File(native_path.h5_path, "r") as file:
        forces = file["forces"]
        assert isinstance(forces, h5py.Dataset)
        assert forces.shape == (input_traj.n_frames, input_traj.n_atoms, 3)
        forces = forces[:, atoms_idx, :]
        if stride > 1:
            forces = forces[:: stride, :, :]
            out_traj = out_traj[:: stride]
        assert forces.shape == (out_traj.n_frames, out_traj.n_atoms, 3)
        return NativeTraj(out_traj, forces, native_path.h5_path)

def hdf5_to_traj(topology: mdtraj.core.topology.Topology, data: Frames) -> mdtraj.Trajectory:
    return mdtraj.Trajectory(
        xyz=data.coordinates,
        topology=topology,
        time=data.time,
        unitcell_lengths=data.cell_lengths,
        unitcell_angles=data.cell_angles,
    )
