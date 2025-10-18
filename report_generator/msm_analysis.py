# report_generator/msm_analysis.py
import numpy
import numpy.typing
import mdtraj
import logging
import itertools
import deeptime
import preprocess
from typing import Optional
from pathlib import Path
from dataclasses import dataclass

from .traj_loading import apply_cg
from .tica_plots import DimensionalityReduction, calc_atom_distance

@dataclass
class MsmRmsdStatistics:
    num_tica_components_used: int
    native_rmsd_mean: float
    native_rmsd_stddev: float
    native_macro_prob: float
    microstate_kmeans: deeptime.clustering.KMeans
    macrostate_assigments: list[int | None]
    native_macrostate_id: int


def get_expiremental_structure(rmsd_dir: Path, protein_name: str, prior_params: dict) -> mdtraj.Trajectory:
    experimental_structure_path = rmsd_dir.joinpath(f"{protein_name}.pdb")
    exp_struct_aa = mdtraj.load(experimental_structure_path)
    exp_structure = apply_cg(
        exp_struct_aa,
        preprocess.prior_types[prior_params["prior_configuration_name"]]().select_atoms(exp_struct_aa.top))

    return exp_structure
    
    
def do_msm_analysis(
        protein_name: str,
        input_trajs: list[mdtraj.Trajectory],
        component_analysis_model: DimensionalityReduction,
        prior_params: dict,
        rmsd_dir: Path
) -> MsmRmsdStatistics:
    """
        Calculate metrics for Table 2 of the paper:
        - Identify native macrostate
        - Calculate equilibrium probability
        - Calculate mean and min RMSD to experimental structure
    """
    logging.info(f"generating MSM model")

    
    exp_structure = get_expiremental_structure(rmsd_dir, protein_name, prior_params)

    # 1. Load experimental structure
    logging.info(f"Creating reference structure from model for {protein_name}")

    # nvm some of them have like ~20??
    # assert exp_structure.n_frames == 1 #native structure should only have 1 frame
        
    test_top = input_trajs[0].top
    assert test_top is not None
    assert exp_structure.top is not None
    exp_atoms = list(exp_structure.top.atoms)
    test_atoms = list(test_top.atoms)
    assert len(exp_atoms) == len(test_atoms), "CRYSTAL TOPOLOGY MISMATCH"
        
    #TODO make sure the topologies actually match, they don't right now because the crystal structure doesn't have bonds in its pdb data but other than that they should match
    # for exp_atom, test_atom in zip(exp_atoms, test_atoms):
    #     assert exp_atom == test_atom, "CRYSTAL TOPOLOGY MISMATCH"


    # percent_native, rmsd = calc_rmsd_and_contacts([x.trajectory for x in native_trajs], exp_structure)
    
    # Also write this structure to a PDB file for future reference
    # exp_structure_out_path = os.path.join(output_dir, f"{protein_name}_reference_structure.pdb")
    # exp_structure.save(exp_structure_out_path)
    # logging.info(f"Saved reference structure to {exp_structure_out_path}")
        
    
    # 2. Project trajectories onto TICA/PCA space
    # model_atom_distances = [calc_atom_distance(t.trajectory) for t in model_trajs]
    # model_projected = component_analysis_model.decompose(model_atom_distances)


    NUM_TICAS_KEEP = 10
    native_atom_distances = [calc_atom_distance(t) for t in input_trajs]
    native_projected = component_analysis_model.decompose(native_atom_distances)
    native_projected_dofs = [x[:, :NUM_TICAS_KEEP] for x in native_projected]
    
    # 3. Build MSM and extract macrostates 
    # Setup clustering
    # model_projected_concat = numpy.concatenate(model_projected)
    native_projected_concat = numpy.concatenate(native_projected_dofs)


    N_KMEANS_CLUSTERS = 100
    native_clustering: deeptime.clustering.KMeans = deeptime.clustering.KMeans(n_clusters=N_KMEANS_CLUSTERS, max_iter=10000).fit(native_projected_concat)
        
    # Create discrete trajectories
    native_dtrajs = [native_clustering.transform(traj) for traj in native_projected_dofs]
    
    # Build MSMs
    native_msm: Optional[deeptime.markov.msm.MarkovStateModelCollection] = \
    deeptime.markov.msm.MaximumLikelihoodMSM().fit_from_discrete_timeseries(
        native_dtrajs, lagtime=1).fetch_model()

    assert native_msm is not None
        
    # PCCA+ clustering
    NUM_MACROSTATES = 5


    macro_offset = 0
    native_macrostate_assignments: list[int | None] = [None for _ in range(N_KMEANS_CLUSTERS)]
    native_microstate_probabilities: list[int | None] = [None for _ in range(N_KMEANS_CLUSTERS)]
    
    for i in range(native_msm.n_connected_msms):
        native_msm.select(i)
        
        component_microstate_ids = native_msm.state_symbols()
        
        num_macros = min(NUM_MACROSTATES, component_microstate_ids.shape[0])
        
        pcca = native_msm.pcca(n_metastable_sets=num_macros)
        native_macrostate_memberships = pcca.memberships
        
        assert native_macrostate_memberships.shape[0] == component_microstate_ids.shape[0]

        stationary_distribution = native_msm.stationary_distribution
        assert stationary_distribution is not None
        
        for i, microstate_id in enumerate(component_microstate_ids):
            native_macrostate_assignments[microstate_id] = numpy.argmax(native_macrostate_memberships[i, :]).astype(int) + macro_offset
            native_microstate_probabilities[microstate_id] = stationary_distribution[i]

        macro_offset += num_macros
        
    # Get macrostate assignments
    
    clusters_per_macrostate: list[numpy.typing.NDArray] = [
        numpy.array([i for i, val in enumerate(native_macrostate_assignments) if val == macro])
        for macro in range(macro_offset)]


    # 5. Calculate metrics for each model type
    native_native_idx, native_rmsds = find_native_state(
        clusters_per_macrostate,
        input_trajs,
        exp_structure,
        native_dtrajs)

    native_rmsd_mean = float(numpy.mean(native_rmsds))
    native_rmsd_stddev = float(numpy.std(native_rmsds))
        
        
    # 6. Get equilibrium probabilities from MSM
    macro_probs: list[int] = [native_microstate_probabilities[x] for x in clusters_per_macrostate[native_native_idx]]
    native_macro_prob = sum(macro_probs)

    return MsmRmsdStatistics(
        num_tica_components_used=NUM_TICAS_KEEP,
        native_macro_prob=native_macro_prob,
        native_rmsd_mean=native_rmsd_mean,
        native_rmsd_stddev=native_rmsd_stddev,
        microstate_kmeans=native_clustering,
        macrostate_assigments=native_macrostate_assignments,
        native_macrostate_id=native_native_idx
    )
        
def find_native_state(
        cluster_per_macrostates: list[numpy.typing.NDArray],
        trajectories: list[mdtraj.Trajectory],
        exp_structure: mdtraj.Trajectory,
        cluster_assignments: list[numpy.typing.NDArray]
) -> tuple[int, numpy.typing.NDArray]:

    min_rmsd = float('inf')
    best_macrostate_idx: int = -1
    all_rmsds: numpy.typing.NDArray | None = None
        
    for i, macrostate_cluster_ids in enumerate(cluster_per_macrostates):
        # Sample frames from this macrostate using the clustering object
        if len(macrostate_cluster_ids) == 0:
            continue
        frames = get_all_frames_from_macrostate(
            macrostate_cluster_ids,
            trajectories,
            cluster_assignments)
        
        # Calculate RMSD to experimental structure
        rmsds: numpy.typing.NDArray = mdtraj.rmsd(frames, exp_structure)
        
        # Ensure rmsds is a proper array, not a slice
        if len(rmsds) > 0:
            current_min = numpy.mean(rmsds)
            if current_min < min_rmsd:
                min_rmsd = current_min
                best_macrostate_idx = i
                all_rmsds = rmsds.copy()  # Make a copy to avoid reference issues
                
                # If we didn't find any valid macrostates, return sensible defaults
    if best_macrostate_idx == -1:
        logging.error("No valid native state found")
        assert False

    assert all_rmsds is not None
    return best_macrostate_idx, all_rmsds


def get_all_frames_from_macrostate(
        macrostate_cluster_ids: numpy.typing.NDArray,
        trajectories: list[mdtraj.Trajectory],
        cluster_assignments: list[numpy.typing.NDArray]
) -> mdtraj.Trajectory:
    """
    Sample frames from a macrostate according to the paper's methodology.
    
    Parameters:
    -----------
    macrostate_microstates : array
        Indices of microstates belonging to the macrostate
    trajectories : list of md.Trajectory
        Original trajectory data
    clustering : deeptime.clustering object,
        Clustering object used to discretize the trajectories
    projected_trajs : list of arrays,
        Precomputed projected trajectories (e.g., TICA/PCA features)
        
    Returns:
    --------
    frames : md.Trajectory
        Concatenated trajectory containing all frames from the macrostate
    """
    assert len(trajectories) == len(cluster_assignments)
    
    frames_to_extract = []
    for traj_idx, dtraj in enumerate(cluster_assignments):
        frame_indices = numpy.where(numpy.isin(dtraj, macrostate_cluster_ids))[0]
        
        if len(frame_indices) > 0:
            traj_frames = trajectories[traj_idx][frame_indices]
            frames_to_extract.append(traj_frames)
    concat_traj = mdtraj.join(frames_to_extract)
    
    return concat_traj


def calc_rmsd_and_contacts(
        compare_trajs: list[mdtraj.Trajectory],
        native_crystal: mdtraj.Trajectory) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]:
    """
    mostly stolen from daniels notebook

    take in simulated traj and the native confromation and output the percent native contacts vs RMSD
    """
    assert native_crystal.topology is not None

    atoms = range(native_crystal.topology.n_atoms)
    
    all_ca_pairs = numpy.array([i for i in itertools.product(atoms, atoms) if i[0] != i[1]])
    native_dist = mdtraj.compute_distances(native_crystal[0], all_ca_pairs)[0]

    native_pairs = all_ca_pairs[native_dist < 0.8]


    compare_trajs_joined = mdtraj.join(compare_trajs)
    
    compare_percent_native = numpy.sum(mdtraj.compute_distances(
        compare_trajs_joined, native_pairs)<0.8,axis=1)/len(native_pairs)


    native_test_rmsd = mdtraj.rmsd(compare_trajs_joined, native_crystal)

    return compare_percent_native, native_test_rmsd
