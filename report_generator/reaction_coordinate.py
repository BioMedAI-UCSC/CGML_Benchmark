from dataclasses import dataclass
import scipy
import mdtraj
import numpy
import numpy.typing
from pathlib import Path

from report_generator.cache_loading import load_cache_or_make_new

@dataclass
class ReactionCoordKde:
    model: scipy.stats.gaussian_kde
    min_val: float
    max_val: float
    
def calc_reaction_coordinate(trajs: list[mdtraj.Trajectory]) -> numpy.typing.NDArray:
    coords = [mdtraj.compute_distances(traj, [(0, traj.n_residues - 1)]) for traj in trajs]
    out = numpy.concatenate(coords)
    assert out.shape[1] == 1
    out = out.flatten()
    return out

def make_react_coord_kde(trajs: list[mdtraj.Trajectory]) -> ReactionCoordKde:
    reaction_coords = calc_reaction_coordinate(trajs)
    kernel = scipy.stats.gaussian_kde(reaction_coords)
    min_val, max_val = numpy.min(reaction_coords), numpy.max(reaction_coords)
    return ReactionCoordKde(kernel, min_val, max_val)

def get_reaction_coordinate_kde(
        trajs: list[mdtraj.Trajectory],
        protein_name: str,
        cache_path: Path,
        force_cache_regen: bool,
        temperature: int) -> tuple[Path, ReactionCoordKde]:
    cache_filename = cache_path.joinpath(f"{protein_name}_{temperature}K.kde_reaction_coordinate")

    return cache_filename, load_cache_or_make_new(
        Path(cache_filename),
        lambda: make_react_coord_kde(trajs),
        ReactionCoordKde,
        force_cache_regen
    )
    
