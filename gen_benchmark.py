#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pickle
import random
import subprocess
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, Tuple

import numpy as np
import yaml
import mdtraj as md
import torch
from tqdm import tqdm
from scipy.stats import gaussian_kde

from machine_conf import MachineConf, list_machine_configs

from report_generator.cache_loading import load_cache_or_make_new
from report_generator.tica_plots import (
    DimensionalityReduction,
    TicaModel,
    PCAModel,
    generate_tica_model_from_scratch,
    generate_pca_model_from_scratch,
)
from report_generator.traj_loading import (
    load_native_trajs_stride,
    load_model_traj_pickle,
    NativeTrajPath,
    NativeTrajPathH5,
    NativeTrajPathNumpy,
    ModelTraj,
    load_model_traj,
)
from report_generator.reaction_coordinate import get_reaction_coordinate_kde
from report_generator.contact_maps import get_contact_maps
from report_generator.bond_and_angle_analysis import get_bond_angles_cached
from report_generator.msm_analysis import do_msm_analysis, MsmRmsdStatistics
from gen_report import runReport

from module.westpa_helpers import (
    load_all_weights_and_trajs_flat,
    get_topology_from_westpa,
    get_implicit_topology_from_westpa,
    get_traj,
    calculate_component_values,
)

# -----------------------------------------------------------------------------
log = logging.getLogger("benchmark")
logging.basicConfig(level=logging.INFO)

# Native traj stride for training TICA/PCA from native
NATIVE_PATHS_STRIDE = 100


class ComponentAnalysisTypes(Enum):
    TICA = 1
    PCA = 2


# -----------------------------------------------------------------------------
# Resource management: GPUs + RAM pressure semaphore
# -----------------------------------------------------------------------------

class ResourceManager:
    """
    Central place for controlling:
    - GPU allocation across concurrent tasks
    - A semaphore to prevent RAM blowups when loading large native trajs
    """
    def __init__(self, gpu_ids: list[int], max_concurrent_native_loads: int = 6):
        self._gpu_ids = gpu_ids
        self._gpu_available = [True] * len(gpu_ids)
        self._gpu_lock = threading.Lock()
        self._gpu_sem = threading.Semaphore(max(1, len(gpu_ids)))

        # Throttle heavy native loads to avoid OOM.
        self.native_load_sem = threading.Semaphore(max_concurrent_native_loads)

    def acquire_gpu(self) -> tuple[int, int]:
        """
        Returns (pool_index, gpu_id).
        Blocks until a GPU is available.
        """
        self._gpu_sem.acquire()
        with self._gpu_lock:
            for idx, ok in enumerate(self._gpu_available):
                if ok:
                    self._gpu_available[idx] = False
                    return idx, self._gpu_ids[idx]
        self._gpu_sem.release()
        raise RuntimeError("GPU semaphore acquired but none available")

    def release_gpu(self, pool_index: int) -> None:
        with self._gpu_lock:
            self._gpu_available[pool_index] = True
        self._gpu_sem.release()


# -----------------------------------------------------------------------------
# Benchmark input “source” strategy objects (model/traj/WESTPA/rerun)
# -----------------------------------------------------------------------------

class BenchmarkSource(Protocol):
    def describe_output_suffix(self) -> str: ...
    def prepare(self) -> None: ...
    def get_model_trajs_and_pickle_path(
        self,
        *,
        protein: str,
        output_dir: Path,
        resources: ResourceManager,
        temperature: int,
        num_steps: int,
        save_steps: int,
        trajs_per_protein: int,
        starting_poses: list[NativeTrajPath],
    ) -> tuple[list[md.Trajectory], Path, Optional[Path], Optional[np.ndarray]]:
        """
        Returns:
          (mdtraj_list, gen_pickle_path, steady_state_msm_kde_path, westpa_weights_used)
        steady_state_msm_kde_path can be None if not produced.
        westpa_weights_used can be None.
        """
        ...


class ModelSource:
    def __init__(
        self,
        *,
        model_path: Path,
        prior_only: bool,
        prior_nn: Optional[Path],
    ):
        self.model_path = model_path
        self.prior_only = prior_only
        self.prior_nn = prior_nn

        self.checkpoint_path: Optional[Path] = model_path if model_path.suffix == ".pth" else None
        self.model_folder: Path = model_path.parent if self.checkpoint_path else model_path

    def describe_output_suffix(self) -> str:
        return self.model_folder.name

    def prepare(self) -> None:
        return

    def _run_model(
        self,
        *,
        protein: str,
        output_dir: Path,
        gpu_id: int,
        temperature: int,
        num_steps: int,
        save_steps: int,
        starting_points: list[NativeTrajPath],
    ) -> Path:
        """
        Runs simulate.py and returns path to produced pickle.
        """
        traj_path = output_dir / f"{protein}_model_replicas.pkl"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

        model_to_run = self.checkpoint_path if self.checkpoint_path else self.model_folder

        cmd = [
            "./simulate.py",
            str(model_to_run),
            *[x.pdb_top_path for x in starting_points],
            "--temperature", str(temperature),
            "--steps", str(num_steps),
            "--save-steps", str(save_steps),
            "--output", str(traj_path),
        ]
        if self.prior_only:
            cmd.append("--prior-only")
        if self.prior_nn is not None:
            cmd += ["--prior-nn", str(self.prior_nn)]

        log.info("Running model for %s on GPU %s", protein, gpu_id)
        with open(output_dir / f"{protein}.log", "w") as outfile:
            subprocess.run(cmd, check=True, env=env, stdout=outfile, stderr=outfile)

        return traj_path

    def get_model_trajs_and_pickle_path(
        self,
        *,
        protein: str,
        output_dir: Path,
        resources: ResourceManager,
        temperature: int,
        num_steps: int,
        save_steps: int,
        trajs_per_protein: int,
        starting_poses: list[NativeTrajPath],
    ) -> tuple[list[md.Trajectory], Path, Optional[Path], Optional[np.ndarray]]:

        picks = random.choices(starting_poses, k=trajs_per_protein)

        pool_idx, gpu_id = resources.acquire_gpu()
        try:
            pickle_path = self._run_model(
                protein=protein,
                output_dir=output_dir,
                gpu_id=gpu_id,
                temperature=temperature,
                num_steps=num_steps,
                save_steps=save_steps,
                starting_points=picks,
            )
        finally:
            resources.release_gpu(pool_idx)

        model_trajs: list[ModelTraj] = load_model_traj_pickle(pickle_path)
        mdtrajs = [t.trajectory for t in model_trajs]
        return mdtrajs, pickle_path, None, None


class TrajFolderSource:
    def __init__(self, folder: Path):
        self.folder = folder
        self.traj_paths = list(folder.iterdir())

    def describe_output_suffix(self) -> str:
        return self.folder.name

    def prepare(self) -> None:
        return

    def get_model_trajs_and_pickle_path(
        self,
        *,
        protein: str,
        output_dir: Path,
        resources: ResourceManager,
        temperature: int,
        num_steps: int,
        save_steps: int,
        trajs_per_protein: int,
        starting_poses: list[NativeTrajPath],
    ) -> tuple[list[md.Trajectory], Path, Optional[Path], Optional[np.ndarray]]:

        model_trajs: list[ModelTraj] = [load_model_traj(path) for path in self.traj_paths]
        mdtrajs = [x.trajectory for x in model_trajs]

        gen_pickle_path = output_dir / f"{protein}_model_replicas.pkl"
        with open(gen_pickle_path, "wb") as f:
            pickle.dump(dict(mdtraj_list=mdtrajs, topology=None, title=""), f)

        return mdtrajs, gen_pickle_path, None, None


class RerunSource:
    def __init__(self, old_dir: Path):
        self.old_dir = old_dir
        self.proteins_pickles: dict[str, Path] = {}

    def describe_output_suffix(self) -> str:
        return "RERUN_" + self.old_dir.name

    def prepare(self) -> None:
        with open(self.old_dir / "benchmark.json", "r") as f:
            bj = json.load(f)
        proteins_dict: dict[str, dict[str, Any]] = bj["proteins"]
        self.proteins_pickles = {name: Path(v["gen_pickle_path"]) for name, v in proteins_dict.items()}

    def get_model_trajs_and_pickle_path(
        self,
        *,
        protein: str,
        output_dir: Path,
        resources: ResourceManager,
        temperature: int,
        num_steps: int,
        save_steps: int,
        trajs_per_protein: int,
        starting_poses: list[NativeTrajPath],
    ) -> tuple[list[md.Trajectory], Path, Optional[Path], Optional[np.ndarray]]:

        pkl = self.proteins_pickles[protein]
        model_trajs: list[ModelTraj] = load_model_traj_pickle(pkl)
        mdtrajs = [x.trajectory for x in model_trajs]

        gen_pickle_path = output_dir / f"{protein}_model_replicas.pkl"
        with open(gen_pickle_path, "wb") as f:
            pickle.dump(dict(mdtraj_list=mdtrajs, topology=None, title=""), f)

        return mdtrajs, gen_pickle_path, None, None


class WestpaSource:
    """
    WESTPA: loads trajectory segments, optionally downsamples count,
    optionally computes a steady-state Markov state model KDE (what used to be 'do-green').

    The weights are kept aligned with traj list.
    """
    def __init__(
        self,
        trajs_locs: list[Path],
        topology: md.Topology,
        weights: np.ndarray,
        max_trajs_load: Optional[int],
        steady_state_markov_state_model_kde: bool,
        cut: bool,
    ):
        self.trajs_locs = trajs_locs
        self.topology = topology
        self.weights = weights
        self.max_trajs_load = max_trajs_load
        self.steady_state_markov_state_model_kde = steady_state_markov_state_model_kde
        self.cut = cut

    def describe_output_suffix(self) -> str:
        return "WESTPA_" + self.trajs_locs[0].parts[-3]

    def prepare(self) -> None:
        return

    @staticmethod
    def _load_one(args: tuple[Path, md.Topology, bool]) -> md.Trajectory:
        path, topology, cut = args
        return get_traj(path, topology, cut=cut)

    def get_model_trajs_and_pickle_path(
        self,
        *,
        protein: str,
        output_dir: Path,
        resources: ResourceManager,
        temperature: int,
        num_steps: int,
        save_steps: int,
        trajs_per_protein: int,
        starting_poses: list[NativeTrajPath],
    ) -> tuple[list[md.Trajectory], Path, Optional[Path], Optional[np.ndarray]]:

        trajs_locs = self.trajs_locs
        weights = self.weights

        # Downsample number of trajectories if requested (keep weights aligned)
        if self.max_trajs_load is not None and self.max_trajs_load > 0:
            stride = max(1, int(len(trajs_locs) / self.max_trajs_load))
            trajs_locs = trajs_locs[::stride]
            weights = weights[::stride]

        # Parallel disk->traj load
        with ProcessPoolExecutor() as ex:
            args = [(p, self.topology, self.cut) for p in trajs_locs]
            trajs = list(tqdm(ex.map(WestpaSource._load_one, args), total=len(trajs_locs)))

        gen_pickle_path = output_dir / f"{protein}_model_replicas.pkl"
        with open(gen_pickle_path, "wb") as f:
            pickle.dump(dict(mdtraj_list=trajs, topology=None, title=""), f)

        # KDE is computed later in pipeline once dimred is available
        return trajs, gen_pickle_path, None, weights


# -----------------------------------------------------------------------------
# Native paths + helpers
# -----------------------------------------------------------------------------

def did_path_finish_simulating(path: str) -> bool:
    finished_path = os.path.join(path, "simulation", "finished.txt")
    if os.path.isfile(finished_path):
        with open(finished_path) as finished_file:
            had_error = "error" in finished_file.read()
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
        return [make_path(x) for x in sorted(list(filter(did_path_finish_simulating, glob.glob(os.path.join(folder, "*")))))]

    return load_cache_or_make_new(Path(f), load_native_paths, list, force_cache_regen)


def get_top_path(coord_path: str) -> str:
    dir_path = os.path.dirname(coord_path[:-len("_coords.npy")])
    base = os.path.basename(coord_path[:-len("_coords.npy")]) + ".pdb"
    return os.path.join(dir_path, "topology", base)


# -----------------------------------------------------------------------------
# Dimensionality reduction model factory
# -----------------------------------------------------------------------------

class DimRedFactory:
    def __init__(self, machine: MachineConf, force_cache_regen: bool, component_type: ComponentAnalysisTypes, temperature: int):
        self.machine = machine
        self.force_cache_regen = force_cache_regen
        self.component_type = component_type
        self.temperature = temperature

    def load_or_build(self, protein: str, native_paths: list[NativeTrajPath], prior_params: dict) -> tuple[DimensionalityReduction, str]:
        if self.component_type == ComponentAnalysisTypes.TICA:
            cache_file = self.machine.analysis_cache_dir / f"{protein}_{self.temperature}K.tica"
            model = load_cache_or_make_new(
                cache_file,
                lambda: generate_tica_model_from_scratch(native_paths, prior_params, NATIVE_PATHS_STRIDE),
                TicaModel,
                self.force_cache_regen,
            )
            return model, str(cache_file)
        else:
            cache_file = self.machine.analysis_cache_dir / f"{protein}_{self.temperature}K.pca"
            model = load_cache_or_make_new(
                cache_file,
                lambda: generate_pca_model_from_scratch(native_paths, prior_params, NATIVE_PATHS_STRIDE),
                PCAModel,
                self.force_cache_regen,
            )
            return model, str(cache_file)


# -----------------------------------------------------------------------------
# Per-protein pipeline (single place for all repeated steps)
# -----------------------------------------------------------------------------

class ProteinPipeline:
    def __init__(
        self,
        *,
        machine: MachineConf,
        temperature: int,
        force_cache_regen: bool,
        enable_msm_metrics: bool,
        component_type: ComponentAnalysisTypes,
        steady_state_markov_state_model_kde: bool,
    ):
        self.machine = machine
        self.temperature = temperature
        self.force_cache_regen = force_cache_regen
        self.enable_msm_metrics = enable_msm_metrics
        self.dimred_factory = DimRedFactory(machine, force_cache_regen, component_type, temperature)
        self.steady_state_markov_state_model_kde = steady_state_markov_state_model_kde

    def _prior_params_for_source(self, source: BenchmarkSource) -> dict:
        # Preserve your original behavior:
        # - model source reads prior_params.json
        # - otherwise default prior configuration
        if isinstance(source, ModelSource):
            path = source.model_folder / "prior_params.json"
            return json.load(open(path, "r"))
        return {"prior_configuration_name": "CA_Majewski2022_v1"}

    def _jsonable(self, obj: Any) -> Any:
        """
        Convert to JSON-safe structures.
        - Skips md.Topology fields (not serializable)
        - Converts Path -> str
        - Converts dataclasses via __dict__
        """
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: self._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._jsonable(x) for x in obj]
        if is_dataclass(obj):
            out: dict[str, Any] = {}
            for k, v in obj.__dict__.items():
                if isinstance(v, md.Topology):
                    continue
                out[k] = self._jsonable(v)
            return out
        return obj

    def _compute_steady_state_msm_kde(
        self,
        *,
        protein: str,
        output_dir: Path,
        dimred: DimensionalityReduction,
        trajs: list[md.Trajectory],
    ) -> Optional[Path]:
        """
        Computes the "green line" KDE: steady-state distribution from a binned transition matrix
        in component space (what your code previously called do_green).
        """
        if not self.steady_state_markov_state_model_kde:
            return None

        all_kde_data: dict[int, dict[str, np.ndarray]] = {}
        components = [0, 1, 2, 3]
        num_bins = 80

        for comp in components:
            component_values: list[np.ndarray] = []
            for traj in tqdm(trajs, desc=f"{protein} steady-state MSM KDE comp {comp}"):
                assert traj.topology is not None
                ca_atoms = traj.topology.select("name CA")
                traj_ca = traj.atom_slice(ca_atoms)
                values = calculate_component_values(dimred, traj_ca, [comp])
                component_values.append(np.asarray(values[comp]))

            cmin = min(float(np.min(cv)) for cv in component_values)
            cmax = max(float(np.max(cv)) for cv in component_values)
            bins = np.linspace(cmin, cmax, num_bins + 1)
            centers = 0.5 * (bins[:-1] + bins[1:])

            binned = [
                np.clip((num_bins * (cv - cmin) / (cmax - cmin)).astype(int), 0, num_bins - 1)
                for cv in component_values
            ]

            T = np.zeros((num_bins, num_bins), dtype=int)
            for bt in binned:
                for j in range(len(bt) - 1):
                    T[bt[j], bt[j + 1]] += 1

            P = T.astype(np.double)
            row_sums = P.sum(axis=1, keepdims=True)
            P = np.divide(P, row_sums, out=np.zeros_like(P), where=row_sums != 0)

            eigenvalues, eigenvectors = np.linalg.eig(P.T)
            stationary_vec = eigenvectors[:, np.isclose(eigenvalues, 1)]
            if stationary_vec.shape[1] != 1:
                log.warning("Component %s stationary not unique; skipping", comp)
                continue

            stationary = (stationary_vec / np.sum(stationary_vec)).real.flatten()
            kde = gaussian_kde(centers, weights=stationary, bw_method=0.1)

            all_kde_data[comp] = {
                "bin_centers": centers,
                "kde_values": kde(centers),
            }

        kde_save_path = output_dir / f"{protein}_steady_state_msm_kde_data.npy"
        np.save(kde_save_path, all_kde_data)  # type: ignore[arg-type]
        return kde_save_path

    def run_one(
        self,
        *,
        protein: str,
        output_dir: Path,
        source: BenchmarkSource,
        native_paths: list[NativeTrajPath],
        starting_poses: list[NativeTrajPath],
        resources: ResourceManager,
        num_steps: int,
        save_steps: int,
        trajs_per_protein: int,
    ) -> dict:
        prior_params = self._prior_params_for_source(source)

        # 1) Dimred model (cached)
        with resources.native_load_sem:
            dimred, dimred_filename = self.dimred_factory.load_or_build(protein, native_paths, prior_params)

        # 2) Model trajs (or traj folder, or rerun, or WESTPA)
        mdtrajs, gen_pickle_path, steady_state_kde_path, weights_used = source.get_model_trajs_and_pickle_path(
            protein=protein,
            output_dir=output_dir,
            resources=resources,
            temperature=self.temperature,
            num_steps=num_steps,
            save_steps=save_steps,
            trajs_per_protein=trajs_per_protein,
            starting_poses=starting_poses,
        )

        # 3) Native trajs load (RAM heavy)
        with resources.native_load_sem:
            native_trajs, all_native_file_strided = load_native_trajs_stride(
                native_paths,
                prior_params,
                NATIVE_PATHS_STRIDE,
                str(self.machine.analysis_cache_dir),
                protein,
                self.force_cache_regen,
                self.temperature,
            )

            # 4) Optional MSM metrics (cached)
            msm_model_cache_path: Optional[str] = None
            if self.enable_msm_metrics:
                msm_model_cache_path = str(self.machine.analysis_cache_dir / f"MSM_native_trajs_{protein}_{self.temperature}K.pkl")
                _ = load_cache_or_make_new(
                    Path(msm_model_cache_path),
                    lambda: do_msm_analysis(
                        protein,
                        [t.trajectory for t in native_trajs],
                        dimred,
                        prior_params,
                        str(self.machine.experimental_structure_rmsd_dir),
                    ),
                    MsmRmsdStatistics,
                    self.force_cache_regen,
                )

            # 5) Artifacts
            contact_map_filename, _ = get_contact_maps(
                [x.trajectory for x in native_trajs],
                protein,
                output_dir,
                self.force_cache_regen,
                temperature=self.temperature,
            )

            reaction_coord_kde_filename, _ = get_reaction_coordinate_kde(
                [x.trajectory for x in native_trajs],
                protein,
                str(self.machine.analysis_cache_dir),
                self.force_cache_regen,
                self.temperature,
            )

            bond_angles_filename, _, _, _ = get_bond_angles_cached(
                native_trajs,
                protein,
                output_dir,
                self.force_cache_regen,
                temperature=self.temperature,
            )

        # 6) Optional steady-state MSM KDE (for WESTPA or any source if enabled)
        if steady_state_kde_path is None and self.steady_state_markov_state_model_kde:
            steady_state_kde_path = self._compute_steady_state_msm_kde(
                protein=protein, output_dir=output_dir, dimred=dimred, trajs=mdtrajs
            )

        out = {
            "gen_pickle_path": str(gen_pickle_path),
            "steady_state_markov_state_model_kde_data_path": str(steady_state_kde_path) if steady_state_kde_path else None,
            "stationary_filename": None,
            "dimensionality_reduction_model": dimred_filename,
            "native_contact_map": str(contact_map_filename),
            "native_reaction_coordinate_kde": str(reaction_coord_kde_filename),
            "native_bond_angles": str(bond_angles_filename),
            "native_paths": [x.__dict__ for x in native_paths],
            "all_native_file_strided": all_native_file_strided,
            "args": sys.argv,
            "msm_model_cache_path": msm_model_cache_path,
        }

        # Persist weights if present
        if weights_used is not None:
            weights_path = output_dir / f"{protein}_westpa_weights.npy"
            np.save(weights_path, weights_used)
            out["westpa_weights_path"] = str(weights_path)

        return self._jsonable(out)


# -----------------------------------------------------------------------------
# Executors: local thread pool or MPI (mpi4py)
# -----------------------------------------------------------------------------

class Executor(Protocol):
    def run(self, proteins: list[str], fn) -> list[dict]: ...


class LocalThreadExecutor:
    def __init__(self, num_threads: int):
        from multiprocessing.dummy import Pool as ThreadPool
        self._ThreadPool = ThreadPool
        self.num_threads = num_threads

    def run(self, proteins: list[str], fn) -> list[dict]:
        with self._ThreadPool(self.num_threads) as pool:
            return pool.map(fn, proteins)


class MPIExecutor:
    """
    Simple MPI implementation:
    - rank 0 distributes proteins to ranks
    - each rank processes its chunk sequentially
    - results gathered back to rank 0

    Requires: pip install mpi4py
    Run: mpirun -n 4 ./benchmark.py --executor mpi ...
    """
    def __init__(self):
        try:
            from mpi4py import MPI  # type: ignore
        except ImportError as e:
            raise RuntimeError("mpi4py not installed; cannot use --executor mpi") from e
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def run(self, proteins: list[str], fn) -> list[dict]:
        chunks = [proteins[i::self.size] for i in range(self.size)]
        my_chunk = self.comm.scatter(chunks, root=0)

        my_results = [fn(p) for p in my_chunk]
        all_results = self.comm.gather(my_results, root=0)

        if self.rank != 0:
            return []

        flat: list[dict] = []
        for part in all_results:
            flat.extend(part)
        return flat


# -----------------------------------------------------------------------------
# Benchmark Orchestrator
# -----------------------------------------------------------------------------

class BenchmarkApp:
    def __init__(
        self,
        *,
        machine: MachineConf,
        proteins: list[str],
        temperature: int,
        output_dir: Optional[Path],
        use_cache: bool,
        only_gen_cache: bool,
        component_type: ComponentAnalysisTypes,
        enable_msm_metrics: bool,
        resources: ResourceManager,
        source: BenchmarkSource,
        executor: Executor,
        steady_state_markov_state_model_kde: bool,
        run_individual_plots: bool,
    ):
        self.machine = machine
        self.proteins = proteins
        self.temperature = temperature
        self.only_gen_cache = only_gen_cache
        self.force_cache_regen = not use_cache
        self.component_type = component_type
        self.enable_msm_metrics = enable_msm_metrics
        self.resources = resources
        self.source = source
        self.executor = executor
        self.run_individual_plots = run_individual_plots

        self.output_dir = self._resolve_output_dir(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.native_paths: dict[str, list[NativeTrajPath]] = {}
        self.starting_poses: dict[str, list[NativeTrajPath]] = {}

        self.pipeline = ProteinPipeline(
            machine=machine,
            temperature=temperature,
            force_cache_regen=self.force_cache_regen,
            enable_msm_metrics=enable_msm_metrics,
            component_type=component_type,
            steady_state_markov_state_model_kde=steady_state_markov_state_model_kde,
        )

    def _resolve_output_dir(self, output_dir: Optional[Path]) -> Path:
        if output_dir is not None:
            return output_dir

        sim_nr = 1
        flds = list(self.machine.benchmark_outputs_dir.glob("0*"))
        if flds:
            sim_nr = max(int(f.parts[-1][:6]) for f in flds) + 1

        suffix = self.source.describe_output_suffix()
        return self.machine.benchmark_outputs_dir / f"{sim_nr:06d}_{suffix}"

    def _init_native_paths(self) -> None:
        if self.temperature == 350:
            for p in self.proteins:
                path = self.machine.native_data_350k_dir / f"{p}_ca_coords.npy"
                self.native_paths[p] = [NativeTrajPathNumpy(str(path), get_top_path(str(path)))]
                # Preserve your existing behavior: 350K uses 300K data for random starting poses
                self.starting_poses[p] = get_native_paths(str(self.machine.native_data_300k_dir / p), self.force_cache_regen)
        elif self.temperature == 300:
            for p in self.proteins:
                self.native_paths[p] = get_native_paths(str(self.machine.native_data_300k_dir / p), self.force_cache_regen)
            self.starting_poses = self.native_paths
        else:
            raise ValueError("temperature must be 300 or 350")

    def run(
        self,
        *,
        num_steps: int,
        save_steps: int,
        trajs_per_protein: int,
        disable_wandb: bool,
        calc_kl: bool,
        westpa_weights: Optional[np.ndarray],
    ) -> Path:
        self.source.prepare()
        self._init_native_paths()

        def run_one_protein(protein: str) -> dict:
            return self.pipeline.run_one(
                protein=protein,
                output_dir=self.output_dir,
                source=self.source,
                native_paths=self.native_paths[protein],
                starting_poses=self.starting_poses[protein],
                resources=self.resources,
                num_steps=num_steps,
                save_steps=save_steps,
                trajs_per_protein=trajs_per_protein,
            )

        results = self.executor.run(self.proteins, run_one_protein)

        # MPI: only rank 0 gets results; non-root ranks return now.
        if not results:
            return self.output_dir / "benchmark.json"

        benchmarks = {protein: result for protein, result in zip(self.proteins, results)}

        benchmark_file = self.output_dir / "benchmark.json"
        with open(benchmark_file, "w") as f:
            f.write(json.dumps(
                {
                    "proteins": benchmarks,
                    "temperature": self.temperature,
                    "used_cache": not self.force_cache_regen,
                    "model_path": None,
                    "experimental_structure_rmsd_dir": str(self.machine.experimental_structure_rmsd_dir),
                },
                indent=4,
            ))

        if self.only_gen_cache:
            return benchmark_file

        runReport(
            benchmark_file,
            also_plot_locally=True,
            do_rmsd_metrics=self.enable_msm_metrics,
            do_kl_divergence=calc_kl,
            disable_wandb=disable_wandb,
            westpa_weights=westpa_weights,
            plot_individuals=self.run_individual_plots,
        )

        log.info("Saved benchmark results: %s", benchmark_file)
        return benchmark_file


# -----------------------------------------------------------------------------
# CLI: machine config dir selection + per-field overrides
# -----------------------------------------------------------------------------

def load_machine_from_dir(machine_config_dir: Path, machine_name: str) -> MachineConf:
    configs = list_machine_configs(machine_config_dir)
    if machine_name not in configs:
        available = ", ".join(sorted(configs.keys()))
        raise ValueError(f"Unknown --machine '{machine_name}'. Available: {available}")
    return MachineConf.load_json(configs[machine_name])


def apply_machine_overrides_from_args(base: MachineConf, args: argparse.Namespace) -> MachineConf:
    return base.with_overrides(
        native_data_300k_dir=args.native_data_300k_dir,
        native_data_350k_dir=args.native_data_350k_dir,
        analysis_cache_dir=args.analysis_cache_dir,
        benchmark_outputs_dir=args.benchmark_outputs_dir,
        experimental_structure_rmsd_dir=args.experimental_structure_rmsd_dir,
    )


def build_source_from_args(args: argparse.Namespace) -> tuple[BenchmarkSource, Optional[np.ndarray], bool, bool]:
    """
    Returns:
      (source, westpa_weights, run_individual_plots, steady_state_markov_state_model_kde_flag)
    """
    run_individual_plots = True
    westpa_weights: Optional[np.ndarray] = None
    steady_state_flag = bool(args.steady_state_markov_state_model_kde)

    if args.model_path is not None:
        src = ModelSource(model_path=Path(args.model_path), prior_only=args.prior_only, prior_nn=args.prior_nn)
        return src, None, True, steady_state_flag

    if args.old_benchmark_dir is not None:
        return RerunSource(Path(args.old_benchmark_dir)), None, True, steady_state_flag

    # Trajs folder: either plain folder or WESTPA
    assert args.trajs_folder is not None
    args.disable_wandb = True

    if args.westpa_implicit_topology:
        topology = get_implicit_topology_from_westpa(args.trajs_folder)
    else:
        topology = get_topology_from_westpa(args.trajs_folder, ext=args.westpa_traj_segment_extension)
        ca_idx = topology.select("protein")
        topology = topology.subset(ca_idx)

    # WESTPA weights handling
    if args.westpa_weights == "mock_weights":
        log.info("Mocking WESTPA weights: using ones")
        pattern = f"seg.{args.westpa_traj_segment_extension}"
        trajs_paths = glob.glob(os.path.join(args.trajs_folder, "traj_segs", "*", "*", pattern))
        westpa_weights = np.ones(len(trajs_paths), dtype=np.float64)

        src = WestpaSource(
            trajs_locs=[Path(x) for x in trajs_paths],
            topology=topology,
            weights=westpa_weights,
            max_trajs_load=args.max_westpa_trajs_to_load,
            steady_state_markov_state_model_kde=steady_state_flag,
            cut=args.westpa_cut_to_single_frame,
        )
        return src, westpa_weights, True, steady_state_flag

    if args.westpa_weights and args.westpa_weights != "None":
        if args.westpa_weights.endswith(".h5") or args.westpa_weights.endswith(".hdf5"):
            westpa_weights, trajs_paths = load_all_weights_and_trajs_flat(
                args.westpa_weights, args.trajs_folder, ext=args.westpa_traj_segment_extension
            )
            src = WestpaSource(
                trajs_locs=[Path(x) for x in trajs_paths],
                topology=topology,
                weights=westpa_weights,
                max_trajs_load=args.max_westpa_trajs_to_load,
                steady_state_markov_state_model_kde=steady_state_flag,
                cut=args.westpa_cut_to_single_frame,
            )
            return src, westpa_weights, True, steady_state_flag

        if args.westpa_weights.endswith(".npy"):
            raise ValueError("Direct .npy weights are not supported in this branch; use .h5/.hdf5 loader so traj list is aligned.")

        raise ValueError("Invalid --westpa-weights. Use mock_weights or .h5/.hdf5.")

    # Plain trajectory folder benchmark (non-WESTPA)
    return TrajFolderSource(Path(args.trajs_folder)), None, True, steady_state_flag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Machine config directory selection
    p.add_argument(
        "--machine-config-dir",
        type=Path,
        default=Path(__file__).with_name("machines"),
        help="Directory containing machine JSON files (e.g., machines/nersc.json).",
    )
    p.add_argument(
        "--machine",
        type=str,
        default=None,
        help="Machine name (filename stem) from --machine-config-dir, e.g. 'nersc' for machines/nersc.json.",
    )

    # Per-field machine overrides (each individually editable from CLI)
    p.add_argument("--native-data-300k-dir", dest="native_data_300k_dir", type=Path, default=None)
    p.add_argument("--native-data-350k-dir", dest="native_data_350k_dir", type=Path, default=None)
    p.add_argument("--analysis-cache-dir", dest="analysis_cache_dir", type=Path, default=None)
    p.add_argument("--benchmark-outputs-dir", dest="benchmark_outputs_dir", type=Path, default=None)
    p.add_argument("--experimental-structure-rmsd-dir", dest="experimental_structure_rmsd_dir", type=Path, default=None)

    # Benchmark source selection (exactly one)
    p.add_argument("--model-path", default=None)
    p.add_argument("--trajs-folder", type=Path, default=None)
    p.add_argument("--old-benchmark-dir", type=Path, default=None)

    p.add_argument("--temperature", type=int, required=True)
    p.add_argument("--proteins", type=str, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, default=None)

    p.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--only-gen-cache", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--prior-only", default=False, action="store_true")
    p.add_argument("--prior-nn", default=None, type=Path)

    p.add_argument("--gpus", default=None, type=str, help='GPU ids like "0,1,2"')
    p.add_argument("--executor", choices=["local", "mpi"], default="local")
    p.add_argument("--threads", type=int, default=None, help="Local thread count; default = #GPUs")

    p.add_argument("--num-steps", type=int, default=100000)
    p.add_argument("--num-save-steps", type=int, default=1000)
    p.add_argument("--trajs-per-protein", type=int, default=20)

    p.add_argument("--component-analysis-type", type=str, default="TICA", choices=["TICA", "PCA"])
    p.add_argument("--calc-kl-divergence", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--enable-msm-metrics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--disable-wandb", action=argparse.BooleanOptionalAction, default=False)

    # WESTPA options (names made verbose/descriptive)
    p.add_argument("--westpa-weights", default="mock_weights", type=str, help="mock_weights or path to .h5/.hdf5")
    p.add_argument("--westpa-cut-to-single-frame", action=argparse.BooleanOptionalAction, default=False,
                   help="Cut WESTPA trajectories to 1 frame each to save memory.")
    p.add_argument("--westpa-traj-segment-extension", type=str, default="dcd", choices=["dcd", "npz"],
                   help="Segment file extension (seg.dcd | seg.npz).")
    p.add_argument("--max-westpa-trajs-to-load", default=None, type=int,
                   help="Max number of WESTPA trajectory segments to load (downsample by stride).")
    p.add_argument("--westpa-implicit-topology", action=argparse.BooleanOptionalAction, default=False,
                   help="Use implicit topology builder for WESTPA.")
    p.add_argument("--steady-state-markov-state-model-kde", action=argparse.BooleanOptionalAction, default=False,
                   help="Compute steady-state MSM KDE curve in component space (was do-green).")

    args = p.parse_args()

    # enforce exactly one benchmark source
    exactly_one = sum(x is not None for x in [args.model_path, args.trajs_folder, args.old_benchmark_dir]) == 1
    if not exactly_one:
        raise ValueError("Must supply exactly one of: --model-path, --trajs-folder, --old-benchmark-dir")

    return args


def resolve_machine(args: argparse.Namespace) -> MachineConf:
    """
    Load machine config from machines/<name>.json, then apply CLI overrides.
    If --machine is missing, fallback to local gen_benchmark.conf (legacy).
    """
    if args.machine is not None:
        base = load_machine_from_dir(args.machine_config_dir, args.machine)
        return apply_machine_overrides_from_args(base, args)

    # Legacy fallback: local yaml like your old behavior
    script_dir = Path(os.path.realpath(sys.argv[0])).parent
    local_conf = script_dir / "gen_benchmark.conf"
    if not local_conf.exists():
        # If user didn't pass --machine, tell them what machine files exist
        configs = list_machine_configs(args.machine_config_dir)
        available = ", ".join(sorted(configs.keys()))
        raise FileNotFoundError(
            f"--machine not provided and gen_benchmark.conf not found.\n"
            f"Available machines in {args.machine_config_dir}: {available}"
        )

    with open(local_conf, "r") as f:
        d = yaml.safe_load(f)

    base = MachineConf.from_dict(d)
    return apply_machine_overrides_from_args(base, args)


def main() -> None:
    args = parse_args()
    machine = resolve_machine(args)

    # GPUs
    if args.gpus:
        gpu_ids = [int(i) for i in args.gpus.strip().split(",")]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
    if not gpu_ids:
        gpu_ids = [0]  # allow CPU-only environments (simulate.py likely still needs GPU)
    resources = ResourceManager(gpu_ids, max_concurrent_native_loads=6)

    # Component analysis
    component_type = ComponentAnalysisTypes.TICA if args.component_analysis_type == "TICA" else ComponentAnalysisTypes.PCA

    # Source
    source, westpa_weights, run_individual_plots, steady_state_flag = build_source_from_args(args)

    # Executor
    if args.executor == "mpi":
        executor = MPIExecutor()
    else:
        nthreads = args.threads if args.threads is not None else max(1, len(gpu_ids))
        executor = LocalThreadExecutor(nthreads)

    app = BenchmarkApp(
        machine=machine,
        proteins=args.proteins,
        temperature=args.temperature,
        output_dir=args.output_dir,
        use_cache=args.use_cache,
        only_gen_cache=args.only_gen_cache,
        component_type=component_type,
        enable_msm_metrics=args.enable_msm_metrics,
        resources=resources,
        source=source,
        executor=executor,
        steady_state_markov_state_model_kde=steady_state_flag,
        run_individual_plots=run_individual_plots,
    )

    app.run(
        num_steps=args.num_steps,
        save_steps=args.num_save_steps,
        trajs_per_protein=args.trajs_per_protein,
        disable_wandb=args.disable_wandb,
        calc_kl=args.calc_kl_divergence,
        westpa_weights=westpa_weights,
    )


if __name__ == "__main__":
    main()

