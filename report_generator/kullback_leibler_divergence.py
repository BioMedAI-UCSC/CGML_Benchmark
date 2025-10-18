import scipy
import os
import numpy
import numpy.typing
import numpy as np
import ot
import pathlib
from report_generator.cache_loading import load_cache_or_make_new

NUM_TICA_DIMS = 5




# computes the Wasserstein distance between the model and native distributions
def wasserstein(native_kde, model_kde, x_bot, x_top, y_bot, y_top) -> float:
    Xmini, Ymini = numpy.mgrid[x_bot:x_top:25j, y_bot:y_top:25j]
    positionsMini = numpy.vstack([Xmini.ravel(), Ymini.ravel()])
    Z_native_1D_mini = native_kde(positionsMini)
    Z_model_1D_mini = model_kde(positionsMini)

    # Normalize weights (to ensure they sum to 1)
    Z_native_1D_mini /= np.sum(Z_native_1D_mini)
    Z_model_1D_mini /= np.sum(Z_model_1D_mini)

    # # This actually computes the 2D 
    # # w1_tica2D = scipy.stats.wasserstein_distance_nd(u_values=positionsMini.T, v_values=positionsMini.T, u_weights=Z_native_1D_mini, v_weights=Z_model_1D_mini)
    # w1_tica2D = wasserstein_distance_nd_mod(u_values=positionsMini.T, v_values=positionsMini.T, u_weights=Z_native_1D_mini, v_weights=Z_model_1D_mini)
    
    # Compute the cost matrix (e.g., squared Euclidean distances)
    cost_matrix = ot.dist(positionsMini.T, positionsMini.T, metric='euclidean')

    # Compute Wasserstein distance using POT
    w1_tica2D = ot.emd2(Z_native_1D_mini, Z_model_1D_mini, cost_matrix)

    assert isinstance(w1_tica2D, float)
    return w1_tica2D

# computes the Wasserstein distance between the model and native distributions
def wasserstein1d(native_kde, model_kde, x_bot, x_top) -> float:
    """
    no idea if this is correct -- andy
    """
    positions = numpy.mgrid[x_bot:x_top:100j]
    Z_native_1D_mini = native_kde(positions)
    Z_model_1D_mini = model_kde(positions)

    # Normalize weights (to ensure they sum to 1)
    Z_native_1D_mini /= np.sum(Z_native_1D_mini)
    Z_model_1D_mini /= np.sum(Z_model_1D_mini)

    # Compute the cost matrix (e.g., squared Euclidean distances)
    cost_matrix = ot.dist(positions[..., None], positions[..., None], metric='euclidean')
    # Compute Wasserstein distance using POT
    w1_tica1D = ot.emd2(Z_native_1D_mini, Z_model_1D_mini, cost_matrix)
    assert isinstance(w1_tica1D, float)
    return w1_tica1D


# computes KL(ground-truth|model) = \sum_{x \in P(GT)} log(GT(x) / M(x)) / |x|
# where P(GT) is the ground-truth (=native) distribution and P(M) is the model distribution
# Raz: I just checked the code below, and this computes the KL(model|native), but we should do the other way around KL(native|model).
def kl_div_calc(native_kde, model_kde, positions) -> float:
    model_pdf = model_kde(positions)
    model_pdf = numpy.maximum(model_pdf, 0.000001)
    points2_pdf1 = native_kde(positions)
    points2_pdf1 = numpy.maximum(points2_pdf1, 0.000001)#replace 0's with this so there isn't divide by zero errors
    #KL(A|B) = ∑∀x P(x)*log(P(x)/Q(x))
    #but its already sampled so if X is the sample then
    #KL(A|B) = ∑ₓ∈X log(P(x)/Q(x)) / |X|
    kullback_leibler_divergence = numpy.sum(numpy.log(points2_pdf1/model_pdf))/len(model_pdf)

    return kullback_leibler_divergence


def kullback_leibler_divergence_tica_1d(model_tica_poses: numpy.typing.NDArray, native_points: numpy.typing.NDArray, protein_name: str, cache_path: str, use_cache: bool, temperature: int) -> tuple[str, float]:
    filename, native_kernels = get_native_kernels(native_points, protein_name, cache_path, use_cache, temperature)
    native_kernel = native_kernels[0]

    first_component = model_tica_poses[::100, 0].T
    
    model_kernel = scipy.stats.gaussian_kde(first_component)

    kl_div = kl_div_calc(native_kernel, model_kernel, first_component)
    
    return filename, kl_div

def get_native_kernels(
        native_points: numpy.typing.NDArray,
        protein_name: str,
        cache_path: str,
        use_cache: bool,
        temperature: int) -> tuple[str, list[scipy.stats.gaussian_kde]]:
    cache_filename = os.path.join(cache_path, f"{protein_name}_{temperature}K.kde")

    make_new = lambda: [scipy.stats.gaussian_kde(native_points[:, x].T) for x in range(NUM_TICA_DIMS)]
    return cache_filename, load_cache_or_make_new(
        pathlib.Path(cache_filename),
        make_new,
        list[scipy.stats.gaussian_kde],
        not use_cache
    )
