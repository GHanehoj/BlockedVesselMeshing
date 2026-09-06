import math
import numpy as np
from scipy.integrate import quad

def compact_polynomial_integral(delta_tau, tau0, l2, uv, d2, i, t):
    return (((delta_tau*delta_tau - l2)*t*t - 2*(-delta_tau*tau0 - uv)*t + (tau0*tau0 - d2))**(i/2))/((delta_tau*t+tau0)**(i+1))

def normalization_factor(i, sigma):
    if i == 0:
        return 2 * sigma * math.sqrt(1-1/(sigma*sigma))
    else:
        return i/(i+1) * (1-1/(sigma*sigma)) * normalization_factor(i-2, sigma)


def get_integral_at_point(p_a, r_a, p_b, r_b, p_p, sigma):
    # Swap points A and B so we go from low to high.
    if r_a > r_b:
        r_a, r_b = r_b, r_a
        p_a, p_b = p_b, p_a

    tau_0 = r_a
    delta_tau = r_b - r_a

    sigma2 = sigma*sigma
    l2 = np.linalg.norm(p_b-p_a)**2 / sigma2
    d2 = np.linalg.norm(p_p-p_a)**2 / sigma2
    uv = np.dot(p_b-p_a, p_p-p_a) / sigma2

    a =     (l2 - delta_tau * delta_tau) * sigma2
    b = 2 * (uv + delta_tau * tau_0)     * sigma2
    c =     (d2 -     tau_0 * tau_0)     * sigma2

    if b**2-4*a*c < 0.0:
        return 0
    else:
        lim1 = max(0, min(1, (b - math.sqrt(b**2-4*a*c)) / (2*a)))
        lim2 = max(0, min(1, (b + math.sqrt(b**2-4*a*c)) / (2*a)))
    if lim1 == lim2:
        return 0

    vals = quad(lambda x: compact_polynomial_integral(delta_tau, tau_0, l2, uv, d2, 6, x), lim1, lim2)
    val = vals[0]


    return max(-10, min(10, val * np.linalg.norm(p_b-p_a) / normalization_factor(6, sigma)))
