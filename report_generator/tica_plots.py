import scipy #type: ignore
import sklearn
import sklearn.decomposition
import mdtraj #type: ignore
import deeptime #type: ignore
import numpy
from report_generator.traj_loading import native_traj_iter_loader
import itertools
import numpy.typing
from report_generator.traj_loading import NativeTrajPath
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import sklearn.decomposition

LAGTIME=20

# Above this many pair features, fitting TICA's covariance matrix
# (N_pairs × N_pairs × 8 bytes) blows up RAM. For a 3,500-bead system
# that's 290 GB. When we exceed the cap we deterministically subsample
# pairs (same RNG seed → identical subset across native + model
# trajectories, which is required for the projection to be meaningful).
# The default of 50,000 keeps the covariance matrix around 20 GB —
# fits on a fat node, still well-conditioned for TICA.
MAX_PAIR_FEATURES = 50_000
SUBSAMPLE_SEED = 12345


def _choose_pair_subset(n_atoms: int, max_pairs: int = MAX_PAIR_FEATURES) -> numpy.typing.NDArray | None:
    """If n_atoms produces more than max_pairs pair features, return a
    deterministic (seed=SUBSAMPLE_SEED) subset of pair indices into the
    full upper-triangle ordering. Returns None if no subsampling needed.
    """
    n_pairs = n_atoms * (n_atoms - 1) // 2
    if n_pairs <= max_pairs:
        return None
    rng = numpy.random.default_rng(SUBSAMPLE_SEED)
    idx = rng.choice(n_pairs, size=max_pairs, replace=False)
    idx.sort()
    return idx


class DimensionalityReduction(ABC):
    @abstractmethod
    def __init__(
            self,
            native_trajs: list[NativeTrajPath],
            protein_name: str,
            cache_path: str,
            use_cache: bool,
            temperature: int,
            prior_params,
            stride: int):
        pass

    @abstractmethod
    def decompose(self, atom_distances: list[numpy.typing.NDArray]) -> list[numpy.typing.NDArray]:
        pass

    @abstractmethod
    def get_title_name(self) -> str:
        pass

    @abstractmethod
    def get_axis_name(self) -> str:
        pass


@dataclass
class TicaModel(DimensionalityReduction):
    tica_model: deeptime.decomposition.CovarianceKoopmanModel
    kde: scipy.stats.gaussian_kde
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    # When the system was too large for full TICA we subsampled pair
    # features; the model can only transform inputs computed with the
    # same subset. None = full upper-triangle ordering.
    pair_indices: numpy.typing.NDArray | None = None
    n_atoms: int = 0

    def decompose(self, atom_distances: list[numpy.typing.NDArray]) -> list[numpy.typing.NDArray]:
        # Each `x` is shaped (n_frames, n_pairs_full). If we subsampled at
        # fit time, take the same columns now; otherwise pass through.
        if self.pair_indices is not None:
            return [self.tica_model.transform(x[:, self.pair_indices]) for x in atom_distances]
        return [self.tica_model.transform(x) for x in atom_distances]

    def get_title_name(self) -> str:
        return "TICA" + (f" (subsampled {len(self.pair_indices)}/{self.n_atoms*(self.n_atoms-1)//2} pairs)" if self.pair_indices is not None else "")
    def get_axis_name(self) -> str:
        return "TIC"
@dataclass
class PCAModel(DimensionalityReduction):
    pca_model: sklearn.decomposition.IncrementalPCA
    kde: scipy.stats.gaussian_kde
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    pair_indices: numpy.typing.NDArray | None = None
    n_atoms: int = 0

    # self.cache_filename = os.path.join(cache_path, f"{protein_name}_{temperature}K.pca")

    def decompose(self, atom_distances: list[numpy.typing.NDArray]) -> list[numpy.typing.NDArray]:
        if self.pair_indices is not None:
            return [self.pca_model.transform(x[:, self.pair_indices]) for x in atom_distances]
        return [self.pca_model.transform(x) for x in atom_distances]

    def get_title_name(self) -> str:
        return "PCA" + (f" (subsampled {len(self.pair_indices)}/{self.n_atoms*(self.n_atoms-1)//2} pairs)" if self.pair_indices is not None else "")
    def get_axis_name(self) -> str:
        return "PCA comp"


def calc_atom_distance(traj: mdtraj.Trajectory) -> numpy.typing.NDArray:
    """Compute every Cα-Cα (or P-P / cross) pair distance. Output shape
    (n_frames, n_atoms*(n_atoms-1)/2). Caller is responsible for slicing
    columns down with a `pair_indices` array if a DimensionalityReduction
    model was fit with subsampling."""
    pairs = list(itertools.combinations(range(0, traj.n_atoms), 2))
    distances = mdtraj.compute_distances(traj, pairs)
    return distances


def _resolve_pair_indices(native_trajs: list[NativeTrajPath], prior_params, stride: int) -> tuple[numpy.typing.NDArray | None, int]:
    """Peek at the first native trajectory to learn the bead count, then
    decide whether to subsample pair features. Returns (pair_indices, n_atoms).
    """
    # Cheap: only need n_atoms, not the full trajectory.
    probe = next(iter(native_traj_iter_loader(native_trajs, prior_params, stride)))
    n_atoms = probe.trajectory.n_atoms
    del probe
    return _choose_pair_subset(n_atoms), n_atoms


def generate_tica_model_from_scratch(
        native_trajs: list[NativeTrajPath],
        prior_params,
        stride: int
) -> TicaModel:
    """Fit TICA on pair distances. Memory-bounded: when the system has
    more than MAX_PAIR_FEATURES pair features, we deterministically
    subsample so the N×N covariance matrix stays tractable. The chosen
    indices are saved on the returned TicaModel so .decompose() can apply
    the same subset to model trajectories.

    Optimized for memory usage, is not optimal for speed as trajectories
    are fetched from disk multiple times.
    """
    pair_indices, n_atoms = _resolve_pair_indices(native_trajs, prior_params, stride)
    if pair_indices is not None:
        logging.info(
            f"TICA: {n_atoms} beads → {n_atoms*(n_atoms-1)//2} pairs exceeds "
            f"cap {MAX_PAIR_FEATURES}; subsampling to {len(pair_indices)} features"
        )

    estimator = deeptime.decomposition.TICA(lagtime=LAGTIME, dim=None)

    for i, traj in enumerate(native_traj_iter_loader(native_trajs, prior_params, stride)):
        atom_distances = calc_atom_distance(traj.trajectory)
        if pair_indices is not None:
            atom_distances = atom_distances[:, pair_indices]
        logging.info(f"done {i}/{len(native_trajs)}")
        for X, Y in deeptime.util.data.timeshifted_split(atom_distances, lagtime=LAGTIME, chunksize=200):
            estimator.partial_fit((X, Y))
        del traj

    model = estimator.fetch_model()

    native_projected_datas = []
    for traj in native_traj_iter_loader(native_trajs, prior_params, stride):
        d = calc_atom_distance(traj.trajectory)
        if pair_indices is not None:
            d = d[:, pair_indices]
        native_projected_datas.append(model.transform(d))

    kde_2d = scipy.stats.gaussian_kde(numpy.concatenate(native_projected_datas)[:, :2].transpose())

    tica_data = numpy.concatenate(native_projected_datas)
    xmin, xmax = numpy.min(tica_data[:, 0]), numpy.max(tica_data[:, 0])
    ymin, ymax = numpy.min(tica_data[:, 1]), numpy.max(tica_data[:, 1])

    return TicaModel(
        tica_model=model,
        kde=kde_2d,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        pair_indices=pair_indices,
        n_atoms=n_atoms,
    )

def generate_pca_model_from_scratch(
        native_trajs: list[NativeTrajPath],
        prior_params,
        stride: int
) -> PCAModel:
    """Fit IncrementalPCA on pair distances. Memory-bounded: same
    subsampling rule as the TICA path (see _choose_pair_subset).

    Optimized for memory usage, is not optimal for speed as trajectories
    are fetched from disk multiple times.
    """
    pair_indices, n_atoms = _resolve_pair_indices(native_trajs, prior_params, stride)
    if pair_indices is not None:
        logging.info(
            f"PCA: {n_atoms} beads → {n_atoms*(n_atoms-1)//2} pairs exceeds "
            f"cap {MAX_PAIR_FEATURES}; subsampling to {len(pair_indices)} features"
        )

    model = sklearn.decomposition.IncrementalPCA()
    for i, traj in enumerate(native_traj_iter_loader(native_trajs, prior_params, stride)):
        atom_distances = calc_atom_distance(traj.trajectory)
        if pair_indices is not None:
            atom_distances = atom_distances[:, pair_indices]
        logging.info(f"done {i}/{len(native_trajs)}")
        model.partial_fit(atom_distances)
        del traj

    native_projected_datas = []
    for traj in native_traj_iter_loader(native_trajs, prior_params, stride):
        d = calc_atom_distance(traj.trajectory)
        if pair_indices is not None:
            d = d[:, pair_indices]
        native_projected_datas.append(model.transform(d))

    kde_2d = scipy.stats.gaussian_kde(numpy.concatenate(native_projected_datas)[:, :2].transpose())

    pca_data = numpy.concatenate(native_projected_datas)
    xmin, xmax = numpy.min(pca_data[:, 0]), numpy.max(pca_data[:, 0])
    ymin, ymax = numpy.min(pca_data[:, 1]), numpy.max(pca_data[:, 1])

    return PCAModel(
        pca_model=model,
        kde=kde_2d,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        pair_indices=pair_indices,
        n_atoms=n_atoms,
    )

    
