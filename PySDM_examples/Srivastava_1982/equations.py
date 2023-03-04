class Equations:
    def __init__(self, alpha_star, beta_star):
        self._beta_star = beta_star
        if alpha_star:
            amb = alpha_star - beta_star
            self._alpha_star = alpha_star
            self._A = amb / 2 / alpha_star
            self._B = 1 / np.sqrt(
                (0.5 / alpha_star + beta_star / alpha_star)
                + amb**2 / (4 * alpha_star**2)
            )

    def eq12(self):
        return 0.5 + np.sqrt(0.25 + 0.5 / self._alpha_star)

    def eq13(self, M, dt, c, total_volume, total_number_0, rho, x):
        t = dt * x
        tau = c * M * t
        mean_volume_0 = total_volume / total_number_0
        m0 = rho * mean_volume_0
        mean_mass = self._eq13(m0, tau)
        return mean_mass

    def _eq13(self, m0, tau):
        ebt = np.exp(-self._beta_star * tau)
        return m0 * ebt + (1 + 0.5 / self._beta_star) * (1 - ebt)

    def eq14(self):
        return 1 + 0.5 / self._beta_star

    def eq15(self, m):
        return (m - self._A) * self._B

    def eq15_m_of_y(self, y):
        return (y / self._B) + self._A

    def eq16(self, tau):
        return tau * self._alpha_star / self._B

    @staticmethod
    def eq10(M, dt, c, total_volume, total_number_0, rho, x):
        t = dt * x
        tau = c * M * t
        mean_volume_0 = total_volume / total_number_0
        m0 = rho * mean_volume_0
        mean_mass = m0 + tau / 2
        return mean_mass
