import numpy as np
import numpy.typing as npt
from typing import Callable


class IntegrationTable:
    """
    This class contains functionality for recomputing convolution integrals. The init function allocates and
    pre-computes all values into arrays that can later be used for table lookups. Further, it provides an integrate
    function that linear interpolate into the pre-computed table.
    """

    def __init__(self, N: int, W: float, K: Callable[[npt.ArrayLike], npt.ArrayLike]):
        """
        Initialize the Kernel Integration Table.

        :param N:   Number of entries to use in the integration table (aka samples on u-axis).
        :param W:   The finite width/support radius of the kernel.
        :param K:   The kernel type function
        """
        self.N = N  # The number of sample points to use.
        self.W = W  # The support radius of the Kernel.
        self.K = K  # The kernel function to be used.
        self.values = np.linspace(-W, 0, N)  # The sample values.
        self.dv = W / (N - 1)  # Spacing between to samples.
        self.kernel = K(self.values)  # The kernel value at each sample.
        self.integral = np.zeros_like(self.values)  # The integrated kernel curve upto the sample point.
        # Compute the area integral using the Trapezoidal integration rule.
        for i in range(1, N):
            self.integral[i] = self.integral[i - 1] + (self.kernel[i] + self.kernel[i - 1])
        self.integral *= self.dv / 2.0
        self.C = 2.0 * self.integral[-1]  # Total area under the Kernel curve

    def integrate(self, v):
        """
        Compute the value of the integral table function

        $$ A(v) = \int_{-\infty}^{v} K(u) du $$

        :param v:  The upper bound value for which we wish to know the integral
                   for. This has to be non-positive value.
        :return:   The value of the integral.
        """
        # We locate the two sample values that are sandwiched between the v-value,
        # and then we linearly interpolate between those values.
        w = (v + self.W) / self.dv
        i = int(w) if isinstance(w, float) else w.astype(int)
        j = i + 1
        i = np.clip(i, 0, self.N - 1)
        j = np.clip(j, 0, self.N - 1)
        t = (w - i)
        interpolated_integral = self.integral[i] + t * (self.integral[j] - self.integral[i])
        return interpolated_integral


def compute_integration_bounds(alpha, W, L):
    """
    We make a convenient reformulation when computing the integral

    $$
    f(alpha) = int_0^L K(u-alpha,W) du
    $$

    where $K(.,.)$ is some symmetric kernel with a finite kernel support of $W$.

    We will instead solve the equivalent integral

    $$
    f(alpha) = int_v0^v1 K(v,W) dv
    $$

    The trick lies in thinking of the local integration bounds $v0$ and $v1$ as
    being functions of the alpha value. These bounds work like a mask that
    clip out the part of the kernel that overlaps with the line segment
    in the "global" $u$-space.

    The implementation of this function is vectorized, hence it supports the
    case when asking for bounds for multiple alpha-values in one call.

    :param alpha:    The shifted position of the symmetric kernel on the line segment
    :param W:        The finite kernel support (width of the kernel)
    :param L:        The length of the edge
    :return:         Lower and upper local bounds (v0, v1) of the kernel integral.
    """
    v0 = np.clip(-alpha, -W, W)
    v1 = np.clip((-alpha + L), -W, W)
    return v0, v1


def gaussian_kernel(u, a, b):
    """
    Evaluates the value of a Gaussian kernel

    $$
    \mathcal{K}(u) \equiv a \, \exp{- b u^2}
    $$

    :param u:  The value at which the kernel should be evaluated at.
    :param a:  Kernel parameter, relates to normalization of the kernel.
    :param b:  Kernel parameter, relates to the width of the kernel.
    :return:   The value of the kernel at the give $u$-value.
    """
    return a * np.exp(-b * u * u)


def compute_integral(alpha, r, W, L, C, T):
    """
    This function compute the value of the line-integral $$\int_0^L \, K(u-alpha) \, du$$. It
    is assumed that no closed-form solution exist for the antiderivative of the kernel. Hence, we
    outline here shortly the theory needed to make a numerical approximation based on using an
    integration lookup table. The key ingredient we use is the symmetry assumption of the kernel.

    Let us define the area integral-value function as follows
    $$
    A(x)
    \equiv
    \int_{-\infty}^x \, K(u) \, du
    $$
    where $$K(.)$$ is a symmetric kernel. Because the kernel is symmetric we want to save memory and computations by
    exploiting this symmetry. So we will now look at the half-space of $K$ for $x < 0$. Here we define a precomputed
    look-up value function,
    $$
    T(x)
    \equiv
    \int_{-\infty}^x K(u) du;
    \quad
    \text{Assumption} x <= 0
    $$
    This function is only computed once, it will basically just an array with values to lookup in constant time.
    We can now go back and rewrite $$A(x)$$ using the $$T(x)$$ table lookup function that is only
    defined for negative values of $$x$$.
    $$
    A(x)  \equiv
    \begin{cases}
        T(x)    &   for x<= 0;\\
        C - T(-x)  &   for x > 0
    \end{cases}
    $$
    Here C is the value of $$\int_{-\infty}^{\infty} K(u) du$$ that is the total area under the curve of the kernel.

    The area integral-value function is helpful for finding the value of any definite integral of the
    kernel. That is we have,

    $$
    \int_{v0}^{v1} \, K(v, W) \, dv = A(v1) - A(v0)
    $$

    This relation follows from the symmetry of the Kernel

    :param alpha:    The position along the line integral where we want to compute the kernel integral value at.
    :param r:        The radius of the line integral.
    :param W:        The finite width/support radius of the kernel.
    :param L:        The length of the line integral.
    :param C:        The total area of the kernel.
    :param T:        The kernel integration lookup function.
    :return:         The value of the integral.
    """
    v0 = np.clip(-alpha/r, -W, W)
    v1 = np.clip((-alpha + L)/r, -W, W)
    A0 = np.where(v0 <= 0, T(v0), C - T(-v0))
    A1 = np.where(v1 <= 0, T(v1), C - T(-v1))
    return A1 - A0


def create_kernel(kernel_type="oeltze.preim"):
    """
    This auxiliary function creates a convolution function.

    :param kernel_type:       String to specify the type of kernel to use. Can be Gaussian, Blommenthal or Vessel.
    :return:                  The generated kernel function.
    """
    W = a = b = None
    if kernel_type == "Gaussian":
        sigma = 0.25
        a = 1.0 / (np.sqrt(np.pi * 2.0) * sigma)
        b = 1.0 / (2.0 * sigma)
        W = 15 * sigma
        isovalue = 3 * sigma
    elif kernel_type == "Blommenthal":
        a = 1.0
        b = np.log(2.0)
        W = 10.0
        isovalue = 1.0 / 2.0
    elif kernel_type == "oeltze.preim":
        a = np.sqrt(np.pi/np.log(32)) # Normalize the kernel for more correct radii.
        b = 5.0 * np.log(2.0)
        W = 1.5
        isovalue = 1.0 / 32.0
    else:
        raise ValueError("Unrecognized kernel_type: " + kernel_type)
    K = lambda v: gaussian_kernel(v, a, b)  # The kernel we want to use
    N = 10000  # Number of sample points to use in numerical integration
    table = IntegrationTable(N, W, K)  # Pre-computed integration table
    T = lambda x: table.integrate(x)  # Pre-computed Kernel Integration function
    C = table.C  # Total area under the curve
    # Our final convolution function
    kernel = lambda alpha, beta, r, L: r*compute_integral(alpha, r, W, L, C, T) * K(beta/r)
    return kernel, W, isovalue
