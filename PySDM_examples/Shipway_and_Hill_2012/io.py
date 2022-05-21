import os

import numpy as np
from pyevtk.hl import pointsToVTK
from pyevtk.vtk import VtkGroup
from scipy.io.netcdf import netcdf_file


class NetCDFExporter_1D:
    def __init__(self, data, settings, simulator, filename):
        self.data = data
        self.settings = settings
        self.simulator = simulator
        self.vars = None
        self.filename = filename
        self.XZ = "Z"
        self.nz_wo_reservoir = int(self.settings.z_max / self.settings.dz)
        self.n_save_spec = len(self.settings.save_spec_and_attr_times)

    def _write_settings(self, ncdf):
        for setting in dir(self.settings):
            setattr(ncdf, setting, getattr(self.settings, setting))

    def _create_dimensions(self, ncdf):
        ncdf.createDimension("time", self.settings.nt + 1)
        ncdf.createDimension("height", self.nz_wo_reservoir)

        if self.n_save_spec != 0:
            ncdf.createDimension("time_save_spec", self.n_save_spec)
            for name, instance in self.simulator.particulator.products.items():
                if len(instance.shape) == 2:
                    dim_name = name.replace(" ", "_") + "_bin_index"
                    ncdf.createDimension(dim_name, self.settings.number_of_bins)

    def _create_variables(self, ncdf):
        self.vars = {}
        self.vars["time"] = ncdf.createVariable("time", "f", ["time"])
        self.vars["time"][:] = self.settings.dt * np.arange(self.settings.nt + 1)
        self.vars["time"].units = "seconds"

        self.vars["height"] = ncdf.createVariable("height", "f", ["height"])
        self.vars["height"][:] = self.settings.dz * (
            1 / 2 + np.arange(self.nz_wo_reservoir)
        )
        self.vars["height"].units = "metres"

        if self.n_save_spec != 0:
            self.vars["time_save_spec"] = ncdf.createVariable(
                "time_save_spec", "f", ["time_save_spec"]
            )
            self.vars["time_save_spec"][:] = self.settings.save_spec_and_attr_times
            self.vars["time_save_spec"].units = "seconds"

            for name, instance in self.simulator.particulator.products.items():
                if len(instance.shape) == 2:
                    label = name.replace(" ", "_") + "_bin_index"
                    self.vars[label] = ncdf.createVariable(label, "f", (label,))
                    self.vars[label][:] = np.arange(1, self.settings.number_of_bins + 1)

        for name, instance in self.simulator.particulator.products.items():
            if name in self.vars:
                raise AssertionError(
                    f"product ({name}) has same name as one of netCDF dimensions"
                )

            n_dimensions = len(instance.shape)
            if n_dimensions == 1:
                dimensions = ("height", "time")
            elif n_dimensions == 2:
                dim_name = name.replace(" ", "_") + "_bin_index"
                if self.n_save_spec == 0:
                    continue
                if self.n_save_spec == 1:
                    dimensions = ("height", f"{dim_name}")
                else:
                    dimensions = ("height", f"{dim_name}", "time_save_spec")
            else:
                raise NotImplementedError()

            self.vars[name] = ncdf.createVariable(name, "f", dimensions)
            self.vars[name].units = instance.unit

    def _write_variables(self):
        for var in self.simulator.particulator.products.keys():
            n_dimensions = len(self.simulator.particulator.products[var].shape)
            if n_dimensions == 1:
                self.vars[var][:, :] = self.data[var][-self.nz_wo_reservoir :, :]
            elif n_dimensions == 2:
                if self.n_save_spec == 0:
                    continue
                if self.n_save_spec == 1:
                    self.vars[var][:, :] = self.data[var][-self.nz_wo_reservoir :, :, 0]
                else:
                    self.vars[var][:, :, :] = self.data[var][
                        -self.nz_wo_reservoir :, :, :
                    ]
            else:
                raise NotImplementedError()

    def run(self):
        with netcdf_file(self.filename, mode="w") as ncdf:
            self._write_settings(ncdf)
            self._create_dimensions(ncdf)
            self._create_variables(ncdf)
            self._write_variables()


class VTKExporter_1D:
    def __init__(
        self, settings, data, path=".", verbose=False, exclude_particle_reservoir=True
    ):
        assert len(data["cell origin"].shape) == 1

        self.settings = settings
        self.data = data
        self.path = os.path.join(path, "sd_attributes")
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

        self.verbose = verbose
        self.exclude_particle_reservoir = exclude_particle_reservoir

        self.num_len = len(str(settings.t_max))
        self.exported_times = {}

    def _write_pvd(self):
        pvd_attributes = VtkGroup(self.attributes_file_path)
        for k, v in self.exported_times.items():
            pvd_attributes.addFile(k + ".vtu", sim_time=v)
        pvd_attributes.save()

    def _export_attributes(self, time):
        path = self.attributes_file_path + "_time" + self.add_leading_zeros(time)
        self.exported_times[path] = time

        if self.verbose:
            print("Exporting Attributes to vtk, path: " + path)

        if len(self.data["cell origin"].shape) != 1:
            raise NotImplementedError("Simulation domain is not 1 dimensional.")

        payload = {}
        for k in self.data.keys():
            payload[k] = self.data[k].to_ndarray(raw=True)
        if self.exclude_particle_reservoir:
            reservoir_particles_indexes = np.where(payload["cell origin"] < 0)
            for k in payload.keys():
                payload[k] = np.delete(payload[k], reservoir_particles_indexes)

        x = self.settings.dz * (payload["cell origin"] + payload["position in cell"])
        y = np.full_like(x, 0)
        z = np.full_like(x, 0)

        pointsToVTK(path, x, y, z, data=payload)

    def _add_leading_zeros(self, a):
        return "".join(["0" for i in range(self.num_len - len(str(a)))]) + str(a)

    def run(self):
        for time in self.settings.save_spec_and_attr_times:
            self._export_attributes(time)
            self._write_pvd()
