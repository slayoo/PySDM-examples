import numpy as np


class Equations:
    """Equations from Srivastava 1982: "A Simple Model of Particle Coalescence and Breakup"
    (https://doi.org/10.1175/1520-0469(1982)039%3C1317:ASMOPC%3E2.0.CO;2)
    note: all equations assume constant fragment mass"""

    @staticmethod
    def eq6_alpha_star(*, alpha, c, M):
        return alpha / c / M

    @staticmethod
    def eq6_beta_star(*, beta, c):
        return beta / c

    def __init__(self, alpha_star=None, beta_star=None):
        self._beta_star = beta_star
        self._alpha_star = alpha_star
        if alpha_star and beta_star:
            amb = alpha_star - beta_star
            self._A = amb / 2 / alpha_star
            self._B = 1 / np.sqrt(
                (0.5 / alpha_star + beta_star / alpha_star)
                + amb**2 / (4 * alpha_star**2)
            )

    def eq12(self):
        """equilibrium (τ→∞) mean mass under collisions and spontaneous breakup
        (no collisional breakup)
        expressed as a ratio to fragment mass (i.e., dimensionless)"""
        return 0.5 + (0.25 + 0.5 / self._alpha_star) ** 0.5

    def eq13(self, M, dt, c, total_volume, total_number_0, rho, x, frag_mass):
        """mean mass expressed as a ratio to fragment mass as a function of
        dimensionless scaled time (τ) under coalescence and collisional breakup
        (no spontaneous breakup)"""
        t = dt * x
        tau = c * M * t
        mean_volume_0 = total_volume / total_number_0
        m0 = rho * mean_volume_0
        mean_mass = self._eq13(m0 / frag_mass, tau)
        return mean_mass

    def _eq13(self, m0, tau):
        ebt = np.exp(-self._beta_star * tau)
        return m0 * ebt + (1 + 0.5 / self._beta_star) * (1 - ebt)

    def eq14(self):
        """equilibrium (τ→∞) mean mass expressed as a ratio to fragment mass for
        under collisional merging and breakup (no spontaneous breakup)"""
        return 1 + 0.5 / self._beta_star

    def eq15(self, m):
        return (m - self._A) * self._B

    def eq15_m_of_y(self, y):
        return (y / self._B) + self._A

    def eq16(self, tau):
        return tau * self._alpha_star / self._B

    @staticmethod
    def eq10(M, dt, c, total_volume, total_number_0, rho, x, frag_mass):
        """ration of mean mass to fragment size mass as a function of scaled time
        for the case of coalescence only"""
        t = dt * x
        tau = c * M * t
        mean_volume_0 = total_volume / total_number_0
        m0 = rho * mean_volume_0 / frag_mass
        mean_mass = m0 + tau / 2
        return mean_mass
