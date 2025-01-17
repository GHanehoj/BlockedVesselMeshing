import numpy as np

def zero():
    return np.zeros((3,), dtype=np.float64)


def ones():
    return np.ones((3,), dtype=np.float64)


def make(x, y, z):
    return np.array([x, y, z], dtype=np.float64)


def make_vec4(x, y, z, w):
    return np.array([x, y, z, w], dtype=np.float64)


def i():
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def j():
    return np.array([0.0, 1.0, 0.0], dtype=np.float64)


def k():
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def make_orthonormal_vectors(n):
    """
    This function is used to generate orthonormal vectors. It is given one
    input vector (assumed a normal vector) and then it generates two other
    vector: a tangent vector t and a binormal vector b.

    :param n:  The input normal vector.
    :return:   The triplet of t, b, and n vectors as output.
    """
    # First we make sure we have a unit-normal vector.
    n = np.copy(n)
    n /= np.linalg.norm(n)
    # Next we try to find a direction that is sufficiently different
    # from the normal vector n. We use the coordinate axis mostly
    # pointing away from the n-vector.
    #
    # We will use this unit-direction as guess for a tangent
    # direction.
    [nx, ny ,nz] = np.fabs(n)
    if nx <= ny and nx <= nz:
        t = i()
    if ny <= nx and ny <= nz:
        t = j()
    if nz <= nx and nz <= ny:
        t = k()
    # We now generate a binormal vector, we know
    #
    #   n = t x b
    #   t = b x n
    #   b = n x t
    #
    # We idea is simply to use the t-vector we guessed at to generate
    # a vector we know will be orthonormal to the n-vector.
    b = np.cross(n, t, axis=0)
    b /= np.linalg.norm(b)
    # Now we know that n and be are orthonormal vectors, we we
    # can now compute a third orthonormal t-vector.
    t = np.cross(b, n, axis=0)
    return t, b, n


def cross(a, b):
    return np.cross(a, b, axis=0)


def unit(a):
    return a / np.linalg.norm(a, axis=-1)[...,None]

def norm(a):                 #pragma: no cover
    return np.linalg.norm(a) #pragma: no cover 


def max_abs_component(a):
    b = np.fabs(a)
    if b[0] > b[1] and b[0] > b[2]:
        return 0
    if b[1] > b[2]:
        return 1
    return 2


def rand(lower, upper):
    return np.random.uniform(lower, upper, 3)


def less(a, b):
    if a[0] > b[0]:
        return False
    if a[0] < b[0]:
        return True
    # We know now that a0==b0
    if a[1] > b[1]:
        return False
    if a[1] < b[1]:
        return True
    # We know not that a0==b0 and a1==b1
    if a[2] > b[2]:
        return False
    if a[2] < b[2]:
        return True
    # We know now that a0==b0 and a1==b1 and a2==b2
    return False

def greater(a, b):
    if a[0] > b[0]:
        return True
    if a[0] < b[0]:
        return False
    # We know now that a0==b0
    if a[1] > b[1]:
        return True
    if a[1] < b[1]:
        return False
    # We know not that a0==b0 and a1==b1
    if a[2] > b[2]:
        return True
    if a[2] < b[2]:
        return False
    # We know now that a0==b0 and a1==b1 and a2==b2
    return False


def less_than_equal(a, b):
    return not greater(a, b)


def greater_than_equal(a, b):
    return not less(a, b)