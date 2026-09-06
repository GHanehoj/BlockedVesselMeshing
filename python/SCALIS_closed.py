import math
import numpy as np

###
# An attempt at a closed solution to the scalis equations. Does not work currently.
###

def compact_polynomial_convolution(k, i, d, q, t):
    if k == 0:
        if i == 1:
            return 1/d*math.log(d*t+q)
        else:
            return (-1/((i-1)*d))*1/(math.pow((d*t+q), i-1))
    elif i == 1:
        return (-1)**k * (q**k)/(d**(k+1)) * math.log(d*t+q) + np.sum([(-1)**(k+l)/l * (q**(k-l))/(d**(k-l+1)) * t**l for l in range(1, k+1)])
    else:
        return 1/((i-1)*d) * (k * compact_polynomial_convolution(k-1, i-1, d, q, t) - t**k/((d*t+q)**(i-1)))

def normalization_factor(i, sigma):
    if i == 0:
        return 2 * sigma * math.sqrt(1-1/(sigma*sigma))
    else:
        return i/(i+1) * (1-1/(sigma*sigma)) * normalization_factor(i-2, sigma)

trinomial_expansion_3 = [
    (1, 3, 0, 0),
    (3, 2, 1, 0),
    (3, 1, 2, 0),
    (1, 0, 3, 0),
    (3, 0, 2, 1),
    (3, 0, 1, 2),
    (1, 0, 0, 3),
    (3, 1, 0, 2),
    (3, 2, 0, 1),
    (6, 1, 1, 1),
]

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

    if delta_tau < 0.1:
        v1 = (-3/5*l2*lim1**5*(d2*l2-l2*tau_0**2+4*uv**2)+uv*lim1**4*(3*d2*l2-3*l2*tau_0**2+2*uv**2)-lim1**3*(d2-tau_0**2)*(d2*l2-l2*tau_0**2+4*uv**2)+3*uv*lim1**2*(d2-tau_0**2)**2-lim1*(d2-tau_0**2)**3-1/7*l2**3*lim1**7+l2**2*uv*lim1**6)/tau_0**7
        v2 = (-3/5*l2*lim2**5*(d2*l2-l2*tau_0**2+4*uv**2)+uv*lim2**4*(3*d2*l2-3*l2*tau_0**2+2*uv**2)-lim2**3*(d2-tau_0**2)*(d2*l2-l2*tau_0**2+4*uv**2)+3*uv*lim2**2*(d2-tau_0**2)**2-lim2*(d2-tau_0**2)**3-1/7*l2**3*lim2**7+l2**2*uv*lim2**6)/tau_0**7
        return v2-v1
#              (-3/5*l2*x**5*(d2*l2- l2 t_0^2 + 4 u^2) + u x^4 (3 d2 l2 - 3 l2 t_0^2 + 2 u^2) - x^3 (d2 - t_0^2) (d2 l2 - l2 t_0^2 + 4 u^2) + 3 u x^2 (d2 - t_0^2)^2 - x (d2 - t_0^2)^3 - 1/7 l^6 x^7 + l^4 u x^6)/t_0^7 + constant


    convolutions = [0, 0, 0, 0, 0, 0, 0]
    for k in range(7):
        convolutions[k] = compact_polynomial_convolution(k, 7, delta_tau, tau_0, lim2) - compact_polynomial_convolution(k, 7, delta_tau, tau_0, lim1)


    aa = delta_tau*delta_tau-l2
    bb = 2*(delta_tau*tau_0+uv)
    cc = tau_0*tau_0-d2

    val = 0
    for idx in range(10):
        coeff, p1, p2, p3 = trinomial_expansion_3[idx]

        k = 2*p1 + 1*p2 + 0*p3
        val += coeff * pow(aa, p1) * pow(bb, p2) * pow(cc, p3) * convolutions[k]

    return max(-10, min(10, val * np.linalg.norm(p_b-p_a) / normalization_factor(6, sigma)))
