import numpy as np

def randomVector(key, seed, k):
    """
    Generate a random vector by setting a unique seed.

    key : tuple, list, or array of floats
        The key specific to this vector

    seed : float or list of floats or array of floats
        The global seed used to generate a set of random vectors

    k: int
        Length of vector to be generated
    """

    if not np.iterable(key):
        key = [key]

    # input checking
    assert(np.iterable(key))
    assert(np.iterable(seed))

    #  create unique key
    uniqueKey = list(key)
    uniqueKey.extend(seed)
    np.random.seed(uniqueKey)

    # generate random output
    return np.random.rand(k)