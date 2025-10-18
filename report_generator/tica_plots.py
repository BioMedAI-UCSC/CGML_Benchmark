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

    def decompose(self, atom_distances: list[numpy.typing.NDArray]) -> list[numpy.typing.NDArray]:
        return [self.tica_model.transform(x) for x in atom_distances]

    def get_title_name(self) -> str:
        return "TICA"
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

    # self.cache_filename = os.path.join(cache_path, f"{protein_name}_{temperature}K.pca")

    def decompose(self, atom_distances: list[numpy.typing.NDArray]) -> list[numpy.typing.NDArray]:
        return [self.pca_model.transform(x) for x in atom_distances]

    def get_title_name(self) -> str:
        return "PCA"
    def get_axis_name(self) -> str:
        return "PCA comp"

    
def calc_atom_distance(traj: mdtraj.Trajectory) -> numpy.typing.NDArray:
    pairs = list(itertools.combinations(range(0, traj.n_atoms), 2))
    distances = mdtraj.compute_distances(traj, pairs)
    return distances


def generate_tica_model_from_scratch(
        native_trajs: list[NativeTrajPath],
        prior_params,
        stride: int
) -> TicaModel:
    """
        Optimized for memory usage, is not optimal for speed as trajectories are fetched from disk multiple times
        """
    estimator = deeptime.decomposition.TICA(lagtime=LAGTIME, dim=None)

    for i, traj in enumerate(native_traj_iter_loader(native_trajs, prior_params, stride)):
        atom_distances = calc_atom_distance(traj.trajectory)
        logging.info(f"done {i}/{len(native_trajs)}")
        for X, Y in deeptime.util.data.timeshifted_split(atom_distances, lagtime=LAGTIME, chunksize=200):
            estimator.partial_fit((X, Y))
        del traj

    model = estimator.fetch_model()

    native_projected_datas = [model.transform(calc_atom_distance(traj.trajectory)) for traj in native_traj_iter_loader(native_trajs, prior_params, stride)]
    
    kde_2d = scipy.stats.gaussian_kde(numpy.concatenate(native_projected_datas)[:, :2].transpose())

    tica_data = numpy.concatenate(native_projected_datas)
    xmin, xmax = numpy.min(tica_data[:, 0]), numpy.max(tica_data[:, 0])
    ymin, ymax = numpy.min(tica_data[:, 1]), numpy.max(tica_data[:, 1])
        
    return TicaModel(
    model,
    kde_2d,
        xmin,
        xmax,
        ymin,
        ymax)

def generate_pca_model_from_scratch(
        native_trajs: list[NativeTrajPath],
        prior_params,
        stride: int
) -> PCAModel:
    """
        Optimized for memory usage, is not optimal for speed as trajectories are fetched from disk multiple times
        """
    model = sklearn.decomposition.IncrementalPCA()
    for i, traj in enumerate(native_traj_iter_loader(native_trajs, prior_params, stride)):
        atom_distances = calc_atom_distance(traj.trajectory)
        logging.info(f"done {i}/{len(native_trajs)}")
        model.partial_fit(atom_distances)
        del traj

        
    native_projected_datas = [model.transform(calc_atom_distance(traj.trajectory)) for traj in native_traj_iter_loader(native_trajs, prior_params, stride)]

    kde_2d = scipy.stats.gaussian_kde(numpy.concatenate(native_projected_datas)[:, :2].transpose())

    pca_data = numpy.concatenate(native_projected_datas)
    xmin, xmax = numpy.min(pca_data[:, 0]), numpy.max(pca_data[:, 0])
    ymin, ymax = numpy.min(pca_data[:, 1]), numpy.max(pca_data[:, 1])
    
    return PCAModel(
        model,
        kde_2d,
        xmin,
        xmax,
        ymin,
        ymax
    )

    
