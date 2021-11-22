from PySDM.physics import constants as const


class FrozenFraction:
    def __init__(self, *, volume, droplet_volume, total_particle_number):
        self.volume = volume
        self.droplet_volume = droplet_volume
        self.total_particle_number = total_particle_number

    def qi2ff(self, ice_mass_per_volume):
        ice_mass = ice_mass_per_volume * self.volume
        ice_number = ice_mass / (const.rho_w * self.droplet_volume)
        frozen_fraction = ice_number / self.total_particle_number
        return frozen_fraction

    def ff2qi(self, frozen_fraction):
        ice_number = frozen_fraction * self.total_particle_number
        ice_mass = ice_number * (const.rho_w * self.droplet_volume)
        ice_mass_per_volume = ice_mass / self.volume
        return ice_mass_per_volume
