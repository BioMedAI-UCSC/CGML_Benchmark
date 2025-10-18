# This is taken from the notebook: scripts/analysis/bond_and_angle_analysis.ipynb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import mdtraj
import os
import numpy.typing
import glob
import logging
from pathlib import Path
from .traj_loading import NativeTraj
from report_generator.cache_loading import load_cache_or_make_new
from module.westpa_helpers import extend_weights

def load_trajectories(coordinate_files):
    coordinate_list = []
    label_list = []

    ## Alternate version that merges the wildcards
    for cf in coordinate_files:
        label_list.append(os.path.basename(cf))
        batch_traj = []
        for subtraj in glob.glob(cf):
            if subtraj.endswith("npy"):
                coords = np.load(subtraj)
                # Select with a stride that brings the total down to ~10,000
                if len(coords > 10000):
                    coords = coords[::(len(coords)//10000)]
                # Convert to NM to match mdtraj coordinates
                coords = coords/10
                psf_path = glob.glob(os.path.join(os.path.dirname(cf),"../processed/*_processed.psf"))[0]
                traj = mdtraj.Trajectory(coords, topology=mdtraj.load_psf(psf_path))
            else:
                traj = mdtraj.load(subtraj)
            batch_traj.append(traj)
        if len(batch_traj) == 0:
            raise RuntimeError(f"{cf} did not match any files")
        coordinate_list.append(mdtraj.join(batch_traj))

    assert len(coordinate_list) == len(label_list)
    return coordinate_list, label_list


def make_plot_grid(length, row_len=3):
    fig, axes = plt.subplots((length+row_len-1)//row_len, min(row_len, length))
    # Ensure axes is always a 2d array
    if isinstance(axes, matplotlib.axes.Axes):
        axes = np.array([[axes]])
    elif len(axes.shape) == 1:
        axes = axes[np.newaxis, :]
    for i in range(length,len(axes.flat)):
        axes.flat[i].remove()
    fig.set_figheight(6*len(axes))
    fig.set_figwidth(6*len(axes[0]))
    return fig, axes

def calculate_bond_lengths(coordinates: mdtraj.Trajectory) -> numpy.typing.NDArray:
    # Calculate bond lengths for a single carbon alpha chain (or some other simple linear series of bonds)
    assert coordinates.n_chains == 1, "Only single chain proteins are supported"
    index_list = []
    assert coordinates.top is not None
    for chain in coordinates.top.chains:
        a_idx = [i.index for i in chain.atoms]
        for i in range(len(a_idx) - 1):
            index_list.append(a_idx[i:i+2])
    
    return mdtraj.compute_distances(coordinates, index_list, periodic=False)

def calculate_bond_angles(coordinates: mdtraj.Trajectory) -> numpy.typing.NDArray:
    # Calculate angles for a single carbon alpha chain (or some other simple linear series of bonds)
    assert coordinates.n_chains == 1, "Only single chain proteins are supported"
    index_list = []
    assert coordinates.top is not None
    for chain in coordinates.top.chains:
        a_idx = [i.index for i in chain.atoms]
        for i in range(len(a_idx) - 2):
            index_list.append(a_idx[i:i+3])
    return mdtraj.compute_angles(coordinates, index_list, periodic=False)

def calculate_dihedrals(coordinates: mdtraj.Trajectory) -> numpy.typing.NDArray:
    # Calculate angles for a single carbon alpha chain (or some other simple linear series of bonds)
    assert coordinates.n_chains == 1, "Only single chain proteins are supported"
    index_list = []
    assert coordinates.top is not None
    for chain in coordinates.top.chains:
        a_idx = [i.index for i in chain.atoms]
        for i in range(len(a_idx) - 3):
            index_list.append(a_idx[i:i+4])
    result = mdtraj.compute_dihedrals(coordinates, index_list, periodic=False)
    # result[result<0] += 2*np.pi
    return result

def plot_hist(ax, values, bins=350, hist_range=None, label=None):
    # Plots a histogram with the x axis values set to the bin centers
    hist = np.histogram(values, range=hist_range, bins=bins, density=True)
    ax.plot((hist[1][:-1] + hist[1][1:])/2, hist[0], label=label)

def plot_bond_length_angles(ax, values, labels, title, xlabel, colors, westpa_weights=None):
    # We could use Freedman Diaconis to get an "optimal" bin width for comparing distributions, however
    # this will crash (out of memory) if the protein exploded because the value range gets too large.
    # Because the numerical comparisons don't seem very helpful yet I've disabled this calculation for now
    # and set the default number of bins to approximately it's average value. - Daniel
    
    try:
        bins = np.histogram_bin_edges(np.concatenate(values).flatten(), bins=70)
        if bins[-1] - bins[0] > 100:
            raise ValueError(f"max-min={bins[-1] - bins[0]:.2f}")
    except ValueError as exc:
        ax.text(0.5, 0.50, f"Exploded\n({exc})", transform=ax.transAxes, ha='center', weight="bold", color="red")
        return
        
    histograms = []
    
    values_1_array = np.concatenate(values[1], axis=0)
    values_for_bonds = values_1_array.flatten()
    if westpa_weights is not None:
        num_bonds_per_frame = values_1_array.shape[1]

        non_zero_weights = westpa_weights[westpa_weights > 0]
        logging.info(f"non-zero weights for bonds before repeat: {non_zero_weights}")

        weights_for_bonds = np.repeat(westpa_weights, num_bonds_per_frame)
        
        
        logging.debug(f"values_for_bonds shape: {values_for_bonds.shape}")
        logging.debug(f"weights_for_bonds shape: {weights_for_bonds.shape}")
        logging.debug(f"weights for bonds array: {weights_for_bonds}")

        non_zero_weights = weights_for_bonds[weights_for_bonds > 0]
        logging.debug(f"non-zero weights for bonds after repeat: {non_zero_weights}")

        if len(weights_for_bonds) != len(values_for_bonds):
            raise ValueError(f"weights_for_bonds ({len(weights_for_bonds)}) does not match flattened values[1] ({len(values_for_bonds)})")
    else:
        weights_for_bonds = None

    histograms.append(np.histogram(values[0], bins=bins, density=True))
    histograms.append(np.histogram(values_for_bonds, bins=bins, density=True, weights=weights_for_bonds))
    logging.debug(f"histograms[0] shape: {histograms[0][0].shape}, histograms[1] shape: {histograms[1][0].shape}")
    logging.debug(f"histograms[1]: {histograms[1][0]}")

    points_x = (bins[:-1] + bins[1:])/2
    bin_widths = (bins[1:] - bins[:-1])

    # Normalize histograms by bin width to get a probability mass function
    # Note this requires density=True above
    histograms = [(h[0]*bin_widths, h[1]) for h in histograms]
        
    for hist, label, color in zip(histograms, labels, colors):
        ax.plot(points_x, hist[0], label=label, color=color)

    
    ax.set_title(title)
    ax.set_xlabel(xlabel)


def get_bond_angles(trajs: mdtraj.Trajectory) -> tuple[list[numpy.typing.NDArray], list[numpy.typing.NDArray], list[numpy.typing.NDArray]]:
    bond_lengths = [calculate_bond_lengths(i) for i in trajs]
    bond_angles = [calculate_bond_angles(i) for i in trajs]
    dihedrals = [calculate_dihedrals(i) for i in trajs]

    return bond_lengths, bond_angles, dihedrals
    
    
def get_bond_angles_cached(
        native_trajs: list[NativeTraj],
        protein_name: str,
        cache_path: Path,
        force_cache_regen: bool,
        temperature: int) -> tuple[Path, list[numpy.typing.NDArray], list[numpy.typing.NDArray], list[numpy.typing.NDArray]]:

    logging.info("Calculating or Loading Cached Bond Length and Angles")
    strideNative = 50
    cache_filename = cache_path.joinpath(f"{protein_name}_{temperature}K_stride{strideNative}_bond_angles.npy")


    def make_new():
        native_traj_mdtraj: mdtraj.Trajectory = mdtraj.join([t.trajectory for t in native_trajs])
        bond_lengths, bond_angles, dihedrals = get_bond_angles(native_traj_mdtraj[::strideNative])
        return dict(bond_lengths=bond_lengths, bond_angles=bond_angles, dihedrals=dihedrals)
            
    baStruct = load_cache_or_make_new(
        cache_filename,
        make_new,
        dict,
        force_cache_regen
    )
    bond_lengths = baStruct['bond_lengths']
    bond_angles = baStruct['bond_angles']
    dihedrals = baStruct['dihedrals']

    return cache_filename, bond_lengths, bond_angles, dihedrals
