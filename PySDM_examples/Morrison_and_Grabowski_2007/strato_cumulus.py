import numpy as np
from PySDM.physics import si
from PySDM_examples.Morrison_and_Grabowski_2007.common import Common


class StratoCumulus(Common):
    def __init__(self, fastmath, rho_w_max: float):
        super().__init__(fastmath)
        self.th_std0 = 289 * si.kelvins
        self.qv0 = 7.5 * si.grams / si.kilogram
        self.p0 = 1015 * si.hectopascals
        self.rho_w_max = rho_w_max

    def stream_function(self, xX, zZ, _):
        X = self.size[0]
        return - self.rho_w_max * X / np.pi * np.sin(np.pi * zZ) * np.cos(2 * np.pi * xX)

    def rhod_of_zZ(self, zZ):
        p = self.formulae.hydrostatics.p_of_z_assuming_const_th_and_qv(
            self.g, self.p0, self.th_std0, self.qv0, z=zZ * self.size[-1])
        rhod = self.formulae.state_variable_triplet.rho_d(p, self.qv0, self.th_std0)
        return rhod
