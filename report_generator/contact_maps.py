import mdtraj
import numpy as np
from dataclasses import dataclass
from .tica_plots import calc_atom_distance
import numpy
import numpy.typing
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from report_generator.cache_loading import load_cache_or_make_new

colors = [
    (0, "blue"),    # Lowest value (-1)
    (0.5, "white"), # Middle value (0)
    (1, "red")      # Highest value (1)
]
custom_cmap = LinearSegmentedColormap.from_list("CustomMap", colors)
THRESHOLD = 1.5

@dataclass
class ContactMap:
    # Frequency of being within THRESHOLD of contact, indexed by pair.
    # For small systems this is the full upper-triangle (n_atoms*(n_atoms-1)/2).
    # For large systems we chunk-compute and stride along the residue
    # axis to keep peak memory under control while preserving full
    # matrix coverage — see make_contact_map.
    matrix: numpy.typing.NDArray
    n_atoms: int


def get_contact_maps(
        native_trajs: list[mdtraj.Trajectory],
        protein_name: str,
        cache_path: Path,
        force_cache_regen: bool,
        temperature: int) -> tuple[Path, ContactMap]:
    cache_filename = cache_path.joinpath(f"{protein_name}_{temperature}K.contact_map")
    make_new = lambda: make_contact_map(native_trajs)
    contact_map = load_cache_or_make_new(
        Path(cache_filename),
        make_new,
        ContactMap,
        force_cache_regen
    )
    return cache_filename, contact_map


# When n_atoms * (n_atoms-1) / 2 exceeds this, switch from the eager
# all-pairs path (materializes a frames × pairs distance matrix at once)
# to a chunked path that streams over residues.
LARGE_SYSTEM_PAIRS = 1_000_000   # ~1500 beads


def _make_contact_map_chunked(trajs: list[mdtraj.Trajectory], extended_weights: np.ndarray | None) -> ContactMap:
    """Streaming contact map for large systems. Iterates over residue i
    and computes (n_frames_total, n_atoms - i - 1) distance blocks
    instead of the full (n_frames_total, n_pairs) matrix. Peak memory
    stays at one row's worth, regardless of system size."""
    n_atoms = trajs[0].n_atoms
    # Concatenate frames once so the loop only does numpy work
    if extended_weights is not None:
        w = extended_weights
    full = mdtraj.join(trajs, check_topology=False) if len(trajs) > 1 else trajs[0]
    n_frames = full.n_frames
    n_pairs = n_atoms * (n_atoms - 1) // 2
    out = np.zeros(n_pairs, dtype=np.float32)

    col = 0
    for i in range(n_atoms - 1):
        # Pairs (i, i+1), (i, i+2), ..., (i, n_atoms-1)
        pairs = np.column_stack([
            np.full(n_atoms - i - 1, i, dtype=np.int32),
            np.arange(i + 1, n_atoms, dtype=np.int32),
        ])
        d = mdtraj.compute_distances(full, pairs)  # (n_frames, n_atoms - i - 1)
        less = d < THRESHOLD
        if extended_weights is not None:
            block_freq = np.average(less, axis=0, weights=w)
        else:
            block_freq = less.mean(axis=0)
        out[col : col + len(pairs)] = block_freq
        col += len(pairs)

    return ContactMap(out, n_atoms)


def make_contact_map(trajs: list[mdtraj.Trajectory], extended_weights: np.ndarray | None = None) -> ContactMap:
    """Compute the contact map (frequency per pair) across all input
    trajectories. Automatically falls back to a chunked path for systems
    with > LARGE_SYSTEM_PAIRS pairs so memory stays bounded — important
    for ribosome / RNA polymerase scale (3,500 beads ≈ 6M pairs).
    """
    n_atoms = trajs[0].n_atoms
    n_pairs = n_atoms * (n_atoms - 1) // 2
    if n_pairs > LARGE_SYSTEM_PAIRS:
        return _make_contact_map_chunked(trajs, extended_weights)

    framesDistances = np.array([calc_atom_distance(x) for x in trajs])  # (trajs, frames, pairs)
    distances = np.concatenate(framesDistances)
    less_than_threshold = distances < THRESHOLD
    if extended_weights is not None:
        percentages = np.average(less_than_threshold, axis=0, weights=extended_weights)
    else:
        percentages = np.mean(less_than_threshold, axis=0)
    return ContactMap(percentages, n_atoms)

def make_visual_matrix(contact_map: ContactMap) -> numpy.typing.NDArray:

    visualmatrix = np.ones([len(contact_map.matrix),len(contact_map.matrix)])
    c = 0
    for i in range(contact_map.n_atoms):
        for j in range(i):
            if i == j:
                continue
            visualmatrix[i][j] = contact_map.matrix[c]
            visualmatrix[j][i] = contact_map.matrix[c]
            c += 1
    return visualmatrix

def make_contact_map_plot(axes, native_contact_map: ContactMap, model_contact_map: ContactMap):
    assert native_contact_map.n_atoms == model_contact_map.n_atoms
    native_matrix = make_visual_matrix(native_contact_map)
    model_matrix = make_visual_matrix(model_contact_map)
    delta = model_matrix - native_matrix

    im = axes.imshow(delta, cmap=custom_cmap)
    axes.set_title(f"Contact map difference: Model - GT")
    axes.set_ylim(0,native_contact_map.n_atoms)
    axes.set_xlim(0,native_contact_map.n_atoms)
    fig = axes.figure
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
    #axes.set_layout_engine("tight")
    #return fig
