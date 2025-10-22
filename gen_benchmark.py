#!/usr/bin/env python3
import argparse
import os
import sys
import glob
from pathlib import Path
import random
import pickle
from typing import Optional
from dataclasses import dataclass, is_dataclass
from report_generator.cache_loading import load_cache_or_make_new
from report_generator.tica_plots import DimensionalityReduction, TicaModel, PCAModel, generate_tica_model_from_scratch, generate_pca_model_from_scratch
from report_generator.traj_loading import load_native_trajs_stride, load_model_traj_pickle, NativeTrajPath, NativeTrajPathH5, NativeTrajPathNumpy, ModelTraj, load_model_traj
from report_generator.reaction_coordinate import get_reaction_coordinate_kde
from report_generator.contact_maps import get_contact_maps
from report_generator.bond_and_angle_analysis import get_bond_angles_cached
from report_generator.msm_analysis import do_msm_analysis, MsmRmsdStatistics
from gen_report import runReport
from westpa_analysis.westpa_helpers import calculate_component_values
import subprocess
import json
import yaml
import numpy as np
import mdtraj as md

import threading
from multiprocessing.dummy import Pool as ThreadPool
from concurrent.futures import ProcessPoolExecutor

import pickle

import torch
from enum import Enum
from tqdm import tqdm

from scipy.stats import gaussian_kde
from module.westpa_helpers import load_all_weights_and_trajs_flat, get_topology_from_westpa, get_implicit_topology_from_westpa, get_traj, calculate_component_values

def wrapper(args):
    path, topology, cut = args
    return get_traj(path, topology, cut=cut)


class ComponentAnalysisTypes(Enum):
    TICA = 1
    PCA = 2


import logging
logging.basicConfig(
    level=logging.DEBUG,
)

# On Delta, nodes have 256GB of RAM and will be killed due to going out of memory (if running on 6 proteins: bba chignolin homeodomain trpcage wwdomain proteinb). Proteinb consumes the highest RAM, around 200GB.
num_threads: int | None = None
gpu_list: list[int] | None = None
generate_trajs_semaphore: threading.Semaphore | None = None
available_gpus: list[bool] | None = None
find_gpu_mutex: Optional[threading.Lock] = None

def init_gpu_locks(gpu_ids: list[int]):
    # This is really ugly but I don't want to rock the boat too much right now - Daniel
    # FIXME: If you see this and want to replace it with something that doesn't use globals go for it
    global num_threads, gpu_list, generate_trajs_semaphore, available_gpus, find_gpu_mutex
    num_threads = len(gpu_ids)
    gpu_list = gpu_ids
    generate_trajs_semaphore = threading.Semaphore(len(gpu_ids))
    available_gpus = [True for _ in gpu_ids]
    find_gpu_mutex = threading.Lock()

# set the semaphore below to 1 when not striding in load_native_trajs_stride as the threads will run out of RAM memory. Loading the entire native trajs for homeodomain/proteinb takes around 200GB RAM.
load_trajs_semaphore = threading.Semaphore(6) 

# I think we'll have to discard the first frames in each model traj, as they are biasing the model KDE towards the native KDE. To see this effect, look on delta at the results with the following command:
#  imgcat /work/hdd/bbpa/benchmarks/000027_all_12368_cyrusc_081724/*.png (used 100k steps and saved every 1,000 steps
# Look at Model Points in TICA space, where R10 shows points after 10% of steps (I plotted it there the other way around) - and R90 shows all 20 replicas after 90,000 steps, which truly shows the equilibrium distribution of the model. 
# Raz later note: I did the above, it's in report.py I think

# total # of frames is 10,000. stride=10 means => 1000 frames * 859 starting points
NATIVE_PATHS_STRIDE = 100 # only take every N frames in the native trajectories

@dataclass
class RefData:
    data_300_path: Path
    data_350_path: Path
    cache_path: Path
    sims_store_dir: Path
    rmsd_dir: Path

@dataclass
class ModelPath:
    model_path: Path
    prior_only: bool
    prior_nn: Path | None
    num_steps: int
    num_save_steps: int
    trajs_per_protein: int

@dataclass
class OldBenchmarkRerun:
    old_benchmark_dir: Path

@dataclass
class TrajFolder:
    traj_folder: Path

@dataclass
class WestpaFolders:
    trajs_locs: list[Path]
    topology: md.Topology
    max_trajs_load: int | None
    ssmsm: bool
    cut: bool
    
Benchmarkables = ModelPath | TrajFolder | OldBenchmarkRerun | WestpaFolders
@dataclass
class BenchmarkModelPath:
    checkpoint_path: Path | None
    model_folder: Path
    prior_only: bool
    prior_nn: Path | None
    num_steps: int
    num_save_steps: int
    trajs_per_protein: int

@dataclass
class BenchmarkTrajFolder:
    folder: Path
    traj_paths: list[Path]
    
@dataclass
class BenchmarkWestpaFolders:
    trajs_locs: list[Path]
    topology: md.Topology
    max_trajs_load: int | None
    ssmsm: bool
    cut: bool

@dataclass
class BenchmarkOldDir:
    folder: Path
    proteins_pickles: dict[str, Path]

class Benchmark:
    temperature: int
    native_paths: dict[str, list[NativeTrajPath]]
    starting_poses: dict[str, list[NativeTrajPath]]
    only_gen_cache: bool
    proteins: list[str]
    ref_data: RefData
    output_dir: Path
    log_dir: Path
    benchmark_descriptor: BenchmarkModelPath | BenchmarkTrajFolder | BenchmarkOldDir | BenchmarkWestpaFolders
    component_analysis: ComponentAnalysisTypes
    make_table: bool
    def __init__(
            self,
            to_benchmark: Benchmarkables,
            use_cache: bool,
            ref_data: RefData,
            proteins: list[str],
            output_dir_c: Path | None,
            only_gen_cache: bool,
            component_analysis: ComponentAnalysisTypes,
            make_table: bool,
            westpa_weights: np.ndarray | None,
    ) -> None:
        self.component_analysis = component_analysis
        self.make_table = make_table
        self.westpa_weights = westpa_weights

        match to_benchmark:
            case TrajFolder(trajs_folder):
                traj_paths = list(trajs_folder.iterdir())
                self.benchmark_descriptor = BenchmarkTrajFolder(trajs_folder, traj_paths)
            case WestpaFolders(trajs_locs, topology, max_trajs_load, ssmsm, cut):
                self.benchmark_descriptor = BenchmarkWestpaFolders(trajs_locs, topology,
                                                                   max_trajs_load, ssmsm, cut)
            case OldBenchmarkRerun(old_dir):
                with open(os.path.join(old_dir, "benchmark.json"), "r") as f:
                    json_data=f.read()

                benchmark_json: dict = json.loads(json_data)
                proteins_dict: dict = benchmark_json["proteins"]
                
                self.benchmark_descriptor = BenchmarkOldDir(
                    old_dir,
                    {name: Path(value["gen_pickle_path"]) for name, value in proteins_dict.items()})                

                
                
                
                    
        self.force_cache_regen = not use_cache
        self.ref_data = ref_data
        self.proteins = proteins
        self.only_gen_cache = only_gen_cache

        # If there's a trajs folder, we're not benchmarking a model, we're benchmarking trajectories


        # machine = machines[self.ref_data]
        if output_dir_c is not None:
            self.output_dir = output_dir_c 
        else:
            simNr = 1
            flds = list(self.ref_data.sims_store_dir.glob("0*"))
            if len(flds) > 0:
                simNr = max([int(f.parts[-1][:6]) for f in flds]) + 1

            match self.benchmark_descriptor:
                case BenchmarkTrajFolder(folder, _):
                    output_postfix = folder.parts[-1]
                case BenchmarkOldDir(folder, _):
                    output_postfix = "RERUN_" + folder.parts[-1]
                case BenchmarkWestpaFolders(folder, _):
                    output_postfix = "WESTPA_" + folder[0].parts[-1]

            self.output_dir = Path(self.ref_data.sims_store_dir).joinpath('%06d' % simNr + '_' + output_postfix)


        self.log_dir = self.output_dir

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving simulations to {self.log_dir}")

    def benchmark_protein(self, protein_name: str) -> dict:        
        model_output_path: Path | None = None                
        prior_params = {"prior_configuration_name":"CA_Majewski2022_v1"}

        with load_trajs_semaphore:
            native_paths = self.native_paths[protein_name]
            logging.info(f"start loading tica model {protein_name}")
            component_analysis_model: DimensionalityReduction
            component_analysis_cache_filename: str
            match self.component_analysis:
                case ComponentAnalysisTypes.TICA:
                    component_analysis_cache_filename = os.path.join(self.ref_data.cache_path, f"{protein_name}_{self.temperature}K.tica")
                    component_analysis_model = load_cache_or_make_new(
                        Path(component_analysis_cache_filename),
                        lambda: generate_tica_model_from_scratch(
                            native_paths,
                            prior_params,
                            NATIVE_PATHS_STRIDE),
                        TicaModel,
                        self.force_cache_regen
                    )
                case ComponentAnalysisTypes.PCA:
                    component_analysis_cache_filename = os.path.join(self.ref_data.cache_path, f"{protein_name}_{self.temperature}K.pca")
                    component_analysis_model = load_cache_or_make_new(
                        Path(component_analysis_cache_filename),
                        lambda: generate_pca_model_from_scratch(
                            native_paths,
                            prior_params,
                            NATIVE_PATHS_STRIDE),
                        PCAModel,
                        self.force_cache_regen
                    )

            component_analysis_model_filename = component_analysis_cache_filename
            logging.info(f"finished loading tica model {protein_name}")

            logging.info(f"start loading model trajs {protein_name}")

            stationary_filename: str| None = None

            kde_save_path: Path | None = None
            
            match self.benchmark_descriptor:
                case BenchmarkTrajFolder(_, traj_paths):
                    model_trajs: list[ModelTraj] = [load_model_traj(path) for path in traj_paths]
                    # save as pickle files for genReport to work
                    gen_pickle_path = f"{os.path.join(self.output_dir, protein_name)}_model_replicas.pkl"
                    with open(gen_pickle_path, "wb") as f:
                         pickle.dump(dict(mdtraj_list=[x.trajectory for x in model_trajs], topology=None, title=""), f)
                case BenchmarkWestpaFolders(trajs_loc, topology, max_trajs_load, ssmsm, cut):
                    # num_keep = 300
                    # trajs_loc = trajs_loc[:num_keep]
                    # self.westpa_weights = self.westpa_weights[:num_keep]
                    # TODO add param for get_traj() to take in stride length, stride until it all fits in memory

                    assert self.westpa_weights is not None
                    if max_trajs_load is not None:
                        stride = int(len(trajs_loc) / max_trajs_load)
                        trajs_loc = trajs_loc[::stride]
                        self.westpa_weights = self.westpa_weights[::stride]
                    
                    with ProcessPoolExecutor() as executor:
                        args = [(path, topology, cut) for path in trajs_loc]
                        trajs = list(tqdm(executor.map(wrapper, args), total=len(trajs_loc)))

                    all_kde_data = {}
                    tica_components = [0, 1, 2, 3]
                    num_bins = 80
                    if ssmsm:
                        for i, comp in enumerate(tica_components):
                            # Step 1: Calculate component values
                            component_values = []
                            for traj in tqdm(trajs):
                                assert traj.topology is not None
                                ca_atoms = traj.topology.select('name CA')
                                traj_ca = traj.atom_slice(ca_atoms)
                                values = calculate_component_values(component_analysis_model, traj_ca, [comp])
                                component_series = np.array(values[comp])
                                component_values.append(component_series)
                            # Step 2: Bin component values
                            component_min = min([min(cv) for cv in component_values])
                            component_max = max([max(cv) for cv in component_values])
                            component_bins = np.linspace(component_min, component_max, num_bins + 1)
                            bin_centers = 0.5 * (component_bins[:-1] + component_bins[1:])

                            binned_components = [
                                np.clip((num_bins * (cv - component_min) / (component_max - component_min)).astype(int), None, num_bins - 1)
                                for cv in component_values
                            ]

                            transition_matrix = np.zeros((num_bins, num_bins), dtype=int)
                            for binned_traj in binned_components:
                                for j in range(len(binned_traj) - 1):
                                    transition_matrix[binned_traj[j], binned_traj[j + 1]] += 1
                            transition_prob_matrix = transition_matrix.astype(np.double)
                            row_sums = transition_prob_matrix.sum(axis=1, keepdims=True)
                            transition_prob_matrix = np.divide(
                                transition_prob_matrix, row_sums, out=np.zeros_like(transition_prob_matrix), where=row_sums != 0
                            )

                            eigenvalues, eigenvectors = np.linalg.eig(transition_prob_matrix.T)
                            stationary_vector = eigenvectors[:, np.isclose(eigenvalues, 1)]
                            if stationary_vector.shape[1] == 1:
                                stationary_distribution = stationary_vector / np.sum(stationary_vector)
                                stationary_distribution = stationary_distribution.real.flatten()
                                kde = gaussian_kde(bin_centers, weights=stationary_distribution, bw_method=0.1)
                                kde_values = kde(bin_centers)

                                all_kde_data[comp] = {
                                    'bin_centers': bin_centers,
                                    'kde_values': kde_values
                                }
                            else:
                                logging.warning(f"Component {comp} stationary not unique")

                    kde_save_path = self.output_dir.joinpath(f"{protein_name}_kde_data.npy")
                    np.save(kde_save_path, all_kde_data)#pyright: ignore[reportArgumentType]

                    # save as pickle files for genReport to work
                    gen_pickle_path = f"{os.path.join(self.output_dir, protein_name)}_model_replicas.pkl"
                    with open(gen_pickle_path, "wb") as f:
                         pickle.dump(dict(mdtraj_list=trajs, topology=None, title=""), f)
                case BenchmarkOldDir(_, proteins_paths):
                    model_trajs: list[ModelTraj] = load_model_traj_pickle(proteins_paths[protein_name])
                    gen_pickle_path = f"{os.path.join(self.output_dir, protein_name)}_model_replicas.pkl"
                    # duplicate pickle files for genReport to work
                    with open(gen_pickle_path, "wb") as f:
                         pickle.dump(dict(mdtraj_list=[x.trajectory for x in model_trajs], topology=None, title=""), f)

            logging.info(f"finished loading model trajs {protein_name}")

            logging.info(f"started loading native trajs {protein_name}")
            native_trajs, all_native_file_strided = load_native_trajs_stride(native_paths, prior_params, NATIVE_PATHS_STRIDE, self.ref_data.cache_path, protein_name, self.force_cache_regen, self.temperature)
            logging.info(f"finished loading native trajs {protein_name}")

            msm_model_cache_path: str | None = None
            if self.make_table:
                msm_model_cache_path = os.path.join(self.ref_data.cache_path, f"MSM_native_trajs_{protein_name}_{self.temperature}K.pkl")

                msm_model: MsmRmsdStatistics = load_cache_or_make_new(
                    Path(msm_model_cache_path),
                    lambda: do_msm_analysis(
                        protein_name,
                        [t.trajectory for t in native_trajs],
                        component_analysis_model,
                        prior_params,
                        self.ref_data.rmsd_dir),
                    MsmRmsdStatistics,
                    self.force_cache_regen
                )

                del msm_model
                    
                        
            logging.info(f"started making native contact map for {protein_name}")
            contact_map_filename, _, = get_contact_maps([x.trajectory for x in native_trajs], protein_name, self.output_dir, self.force_cache_regen, temperature=self.temperature)
            logging.info(f"finished making native contact map for {protein_name}")
            
            reaction_coord_kde_filename, _ = get_reaction_coordinate_kde([x.trajectory for x in native_trajs], protein_name, self.ref_data.cache_path, self.force_cache_regen, self.temperature)



            logging.info(f"started making bond angles for {protein_name}")
            bond_angles_filename, _, _, _ = get_bond_angles_cached(native_trajs, protein_name, self.output_dir, self.force_cache_regen, temperature=self.temperature)
            logging.info(f"finished making bond angles for {protein_name}")


        def save_asdict_excluding_topologies(obj):
            """Convert dataclass to dict while skipping unserializable fields."""
            if not is_dataclass(obj):
                return obj
            result = {}
            for k, v in obj.__dict__.items():
                if isinstance(v, md.Topology):
                    # Skip topology fields as they are not JSON serializable
                    continue
                result[k] = v
            return result


        benchmark_output = {
            "gen_pickle_path": gen_pickle_path,
            "kde_data_path": kde_save_path,
            "stationary_filename": stationary_filename,
            "tica_model": component_analysis_model_filename,
            "contact_map": contact_map_filename,
            "reaction_coord_kde": reaction_coord_kde_filename,
            "bond_angles_filename": bond_angles_filename,
            "native_paths": [x.__dict__ for x in native_paths],
            "all_native_file_strided": all_native_file_strided,
            "args": sys.argv,
            "benchmark_descriptor": save_asdict_excluding_topologies(self.benchmark_descriptor),
            "msm_model": msm_model_cache_path
        }
        #need to save weights as npy file to make it json serializable
        if self.westpa_weights is not None:
            weights_path = os.path.join(self.output_dir, f"{protein_name}_westpa_weights.npy")
            np.save(weights_path, self.westpa_weights)
            benchmark_output["westpa_weights"] = weights_path

        logging.info(f"finished benchmarking protein {protein_name}")
        return benchmark_output
    
    def runParallel(self) -> Path:
        with ThreadPool(num_threads) as pool:
            logging.info("Launching ThreadPool")
            results = pool.map(self.benchmark_protein, self.proteins)

        # benchmarks = self.buildDict(results)
        benchmarks = {
            protein: result
            for protein, result in zip(self.proteins, results)
        }

        benchmarkFile = self.output_dir.joinpath("benchmark.json")
        with open(benchmarkFile, "w") as f:
            match self.benchmark_descriptor:
                case BenchmarkModelPath(_, model_folder, _, _):
                    model_path = model_folder
                case _:
                    model_path = None

            f.write(json.dumps(dict_str_paths({
                "proteins": benchmarks,
                "temperature": self.temperature,
                "used_cache": not self.force_cache_regen,
                "model_path": model_path,
                "rmsd_dir": self.ref_data.rmsd_dir
            }), indent=4))

        return benchmarkFile
        

class Benchmark350(Benchmark):
    def __init__(
            self,
            to_benchmark: Benchmarkables,
            use_cache: bool,
            ref_data: RefData,
            proteins: list[str],
            output_dir_c: Path | None,
            only_gen_cache: bool,
            component_analysis: ComponentAnalysisTypes,
            make_table: bool,
            westpa_weights: np.ndarray | None = None,
    ) -> None:
        self.temperature = 350
        super().__init__(to_benchmark, use_cache, ref_data, proteins, output_dir_c, only_gen_cache, component_analysis, make_table, westpa_weights)

        self.native_paths = {}
        self.starting_poses = {}
        for p in self.proteins:
            path = os.path.join(self.ref_data.data_350_path, f"{p}_ca_coords.npy")
            self.native_paths[p] = [NativeTrajPathNumpy(path, get_top_path(path))]
            self.starting_poses[p] = get_native_paths(os.path.join(self.ref_data.data_300_path, p), self.force_cache_regen)#todo: 350K uses 300K data for random starting poses



class Benchmark300(Benchmark):
    def __init__(
            self,
            to_benchmark: Benchmarkables,
            use_cache: bool,
            ref_data: RefData,
            proteins: list[str],
            output_dir_c: Path | None,
            only_gen_cache: bool,
            component_analysis: ComponentAnalysisTypes,
            make_table: bool,
            westpa_weights: np.ndarray | None = None,
    ) -> None:
        self.temperature = 300
        super().__init__(to_benchmark, use_cache, ref_data, proteins, output_dir_c, only_gen_cache, component_analysis, make_table, westpa_weights)
        self.native_paths = {}
        for p in proteins:
            self.native_paths[p] = get_native_paths(os.path.join(self.ref_data.data_300_path, p), self.force_cache_regen)
        self.starting_poses = self.native_paths



def did_path_finish_simulating(path: str) -> bool:
    finished_path = os.path.join(path, "simulation", "finished.txt")
    if os.path.isfile(finished_path):
        with open(finished_path) as finished_file:
            had_error = 'error' in finished_file.read()
            return not had_error
    return False

def get_native_paths(folder: str, force_cache_regen: bool) -> list[NativeTrajPath]:
    def make_path(base: str):
        basename = os.path.basename(base)
        h5_path = os.path.join(base, "result", f"output_{basename}.h5")
        pdb_path = os.path.join(base, "processed", f"{basename}_processed.pdb")
        return NativeTrajPathH5(h5_path, pdb_path)

    f = os.path.join(folder, "native_paths.pkl")

    def load_native_paths() -> list[NativeTrajPath]:
        return [make_path(x) for x in
            sorted(list(filter(did_path_finish_simulating, glob.glob(os.path.join(folder, "*")))))]

    return load_cache_or_make_new(
        Path(f),
        load_native_paths,
        list,
        force_cache_regen)

def get_top_path(coord_path: str) -> str:
    dir_path = os.path.dirname(coord_path[:-len("_coords.npy")])
    base = os.path.basename(coord_path[:-len("_coords.npy")]) + ".pdb"
    out = os.path.join(dir_path, "topology", base)
    return out


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--temperature", type=int, help="Temperature in Kelvin of the model")
    arg_parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True, help="Regenerate the cache instead of using previous runs data")
    arg_parser.add_argument("--only-gen-cache", action=argparse.BooleanOptionalAction, default=False, help="Only regenerate the stuff that is cached like the TICA model, will not run the model at all")
    arg_parser.add_argument("--ref-data", type=str, default=None, help="reference data root directory")
    arg_parser.add_argument("--proteins", type=str, default=None, help="Proteins to run benchmark on", nargs="+")
    arg_parser.add_argument("--output-dir", type=Path, default=None, help="Output directory of benchmarks")
    arg_parser.add_argument("--gpus", default=None, type=str, help="List of GPUs to use (e.g. \"0,1,2\")")
    arg_parser.add_argument("--disable-wandb", action=argparse.BooleanOptionalAction, default=False, help="Disable wandb logging")
    arg_parser.add_argument("--trajs-folder", type=Path, default=None, help="Directory containing the trajectories of the proteins")
    arg_parser.add_argument("--component-analysis-type", type=str, default="TICA", choices=["TICA", "PCA"], help="Which type of dimensionality reduction to use")
    arg_parser.add_argument("--old-benchmark-dir", type=Path, default=None, help="Old benchmark directory to re-run on")
    # arg_parser.add_argument("--westpa-dir", type=Path, default=None, help="Westpa output to benchmark")
    arg_parser.add_argument("--calc-kl-divergence", action=argparse.BooleanOptionalAction, default=False,
                            help="Calculate the KL divergence for components in tica space")
    arg_parser.add_argument("--enable-msm-metrics", action=argparse.BooleanOptionalAction, 
                       default=True, help="Disable generation of native macrostate statistics")
    arg_parser.add_argument("--westpa-weights", default="mock_weights", type=str, help="Path to westpa weights file, if used, otherwise will be a mock array of ones")
    arg_parser.add_argument("--westpa-cut", action=argparse.BooleanOptionalAction, default=False, help="Cut WESTPA trajectories to only be 1 frame each to save memory")
    arg_parser.add_argument("--traj-extension", type=str, default="dcd", choices=["dcd", "npz"], help="File type for WESTPA trajectory segments (seg.dcd | seg.npz)")
    arg_parser.add_argument("--do-green", action=argparse.BooleanOptionalAction, default=False, help="Enable MSM kde density as green line")
    arg_parser.add_argument("--max-westpa-trajs", default=None, type=int, help="Max westpa trajs to use")
    arg_parser.add_argument("--westpa-implicit", action=argparse.BooleanOptionalAction, default=False, help="Toggle if running on implicit westpa data, idk why ngl")

    args = arg_parser.parse_args()
    
    westpa_weights = None

    assert ((args.trajs_folder is not None) ^
            (args.old_benchmark_dir is not None)), "Must have exactly one of model, trajectory, or old benchmark"
    

    if args.gpus:
        gpu_ids = [int(i) for i in args.gpus.strip().split(",")]
    else:
        gpu_ids = [*range(torch.cuda.device_count())]
    init_gpu_locks(gpu_ids)

    to_benchmark: Benchmarkables | None = None
    run_individual_plots = True
    if args.trajs_folder is not None:
        args.disable_wandb = True

        if args.westpa_implicit:
            topology = get_implicit_topology_from_westpa(args.trajs_folder)
        else:
            topology = get_topology_from_westpa(args.trajs_folder, ext=args.traj_extension)
                
        if args.westpa_weights == "mock_weights": # mock weights, so we can run the WESTPA branch without impact of weights
            logging.info("Mocking WESTPA weights: using all 0s to force WESTPA branch with no actual weighting")
            pattern = f"seg.{args.traj_extension}"
            trajs_paths = glob.glob(os.path.join(args.trajs_folder, "traj_segs", "*", "*", pattern))
            # make fake weights of ones: one per segment file
            n_trajs = len(trajs_paths)
            westpa_weights = np.ones(n_trajs)
            to_benchmark = WestpaFolders([Path(x) for x in trajs_paths], topology,
                                         args.max_westpa_trajs, args.ssmsm, args.westpa_cut)
        elif args.westpa_weights and args.westpa_weights is not None:
            if args.westpa_weights.endswith(".npy"):
                raise ValueError("WESTPA weights as .npy not implemented yet, please use .h5 or .hdf5")
            elif args.westpa_weights.endswith(".h5") or args.westpa_weights.endswith(".hdf5"):
                westpa_weights, trajs_paths = load_all_weights_and_trajs_flat(args.westpa_weights, args.trajs_folder, ext=args.traj_extension)
                to_benchmark = WestpaFolders([Path(x) for x in trajs_paths], topology,
                                             args.max_westpa_trajs, args.ssmsm, args.westpa_cut)
        else:
            westpa_weights = None
            to_benchmark = TrajFolder(args.trajs_folder)
    elif args.old_benchmark_dir is not None:
        to_benchmark = OldBenchmarkRerun(args.old_benchmark_dir)
    # elif args.westpa_dir is not None:
    #     to_benchmark = WestpaFolder(args.westpa_dir)
    #     run_individual_plots = False
    else:
        assert False

    component_analysis_type: ComponentAnalysisTypes
    match args.component_analysis_type:
        case "TICA":
            component_analysis_type = ComponentAnalysisTypes.TICA
        case "PCA":
            component_analysis_type = ComponentAnalysisTypes.PCA
        case _:
            assert False
            
    if args.ref_data_dir is not None:
        ref_data = RefData(Path(f"{args.ref_data_dir}/data300K"),
                         Path(f"{args.ref_data_dir}/data350K"),
                         Path(f"{args.ref_data_dir}/cache"),
                         Path(f"{args.ref_data_dir}/sims"),
                         Path(f"{args.ref_data_dir}/rmsd"))


    if args.westpa_weights is not None and args.westpa_weights != "mock_weights":
        if args.westpa_weights.endswith(".npy"):
            westpa_weights = np.load(args.westpa_weights)
            logging.info(f"Loaded WESTPA weights from .npy: {args.westpa_weights}")
        elif not (args.westpa_weights.endswith(".h5") or args.westpa_weights.endswith(".hdf5")):
            raise ValueError("Invalid --westpa-weights path. Must be a .npy, .h5, or .hdf5 file.")


    assert to_benchmark is not None, "messy logic above this line messed up" 
    # put the code below into a separate function
    if args.temperature == 350:
        logging.info('Running at 350K')
        benchmark = Benchmark350(to_benchmark, args.use_cache, ref_data, args.proteins, args.output_dir, args.only_gen_cache, component_analysis_type, args.enable_msm_metrics, westpa_weights)
    elif args.temperature == 300:
        logging.info('Running at 300K')
        benchmark = Benchmark300(to_benchmark, args.use_cache, ref_data, args.proteins, args.output_dir, args.only_gen_cache, component_analysis_type, args.enable_msm_metrics, westpa_weights)
    else:
        assert False, "temperature must be either 300 or 350"

    if not args.westpa_weights or args.westpa_weights == "None":
        westpa_weights = None

    benchmarkFile = benchmark.runParallel()
    if args.only_gen_cache:
        return
    runReport(benchmarkFile,
              also_plot_locally=True,
              do_rmsd_metrics=args.enable_msm_metrics,
              do_kl_divergence=args.calc_kl_divergence,
              disable_wandb=args.disable_wandb,
              westpa_weights=westpa_weights,
              plot_individuals=run_individual_plots)

    logging.info(f"saved benchmark resultes \"{benchmarkFile}\"")


def dict_str_paths(d: dict) -> dict:
    keys = d.keys()
    for k in keys:
        if isinstance(d[k], dict):
            d[k] = dict_str_paths(d[k])
        elif isinstance(d[k], list):
            d[k] = list_str_paths(d[k])
        elif isinstance(d[k], Path):
          d[k] = str(d[k])

    return d

def list_str_paths(d: list) -> list:
    for k in range(len(d)):
        if isinstance(d[k], dict):
            d[k] = dict_str_paths(d[k])
        elif isinstance(d[k], list):
            d[k] = list_str_paths(d[k])
        elif isinstance(d[k], Path):
          d[k] = str(d[k])

    return d


if __name__ == "__main__":
    main()
