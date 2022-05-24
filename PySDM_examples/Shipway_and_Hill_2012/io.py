import os

import numpy as np
import vtk
from pyevtk.hl import pointsToVTK
from scipy.io.netcdf import netcdf_file
from vtk.util import numpy_support as VN


class NetCDFExporter_1D:
    def __init__(
        self, data, settings, simulator, filename, exclude_particle_reservoir=True
    ):
        self.data = data
        self.settings = settings
        self.simulator = simulator
        self.vars = None
        self.filename = filename
        self.XZ = "Z"
        self.nz_export = (
            int(self.settings.z_max / self.settings.dz)
            if exclude_particle_reservoir
            else settings.nz
        )
        self.n_save_spec = len(self.settings.save_spec_and_attr_times)

    def _write_settings(self, ncdf):
        for setting in dir(self.settings):
            setattr(ncdf, setting, getattr(self.settings, setting))

    def _create_dimensions(self, ncdf):
        ncdf.createDimension("time", self.settings.nt + 1)
        ncdf.createDimension("height", self.nz_export)

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
        self.vars["height"][:] = self.settings.dz * (1 / 2 + np.arange(self.nz_export))
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
                self.vars[var][:, :] = self.data[var][-self.nz_export :, :]
            elif n_dimensions == 2:
                if self.n_save_spec == 0:
                    continue
                if self.n_save_spec == 1:
                    self.vars[var][:, :] = self.data[var][-self.nz_export :, :, 0]
                else:
                    self.vars[var][:, :, :] = self.data[var][-self.nz_export :, :, :]
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
        self,
        data,
        settings,
        path="./sd_attributes/",
        verbose=False,
        exclude_particle_reservoir=True,
    ):

        self.data = data
        self.settings = settings
        self.path = path
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

        self.verbose = verbose
        self.exclude_particle_reservoir = exclude_particle_reservoir

        self.num_len = len(str(settings.t_max))

    def _export_attributes(self, time_index_and_value):
        time_index = time_index_and_value[0]
        time = time_index_and_value[1]
        path = self.path + "time" + self._add_leading_zeros(time)

        if self.verbose:
            print("Exporting Attributes to vtk, path: " + path)

        payload = {}
        for k in self.data.keys():
            if len(self.data[k][time_index].shape) == 1:
                payload[k] = self.data[k][time_index]
            elif len(self.data[k][time_index].shape) == 2:
                assert self.data[k][time_index].shape[0] == 1
                payload[k] = self.data[k][time_index][0]
            else:
                raise NotImplementedError("Shape of data array is not recognized.")

        z = (
            self.settings.dz * (payload["cell origin"] + payload["position in cell"])
            - self.settings.particle_reservoir_depth
        )

        if self.exclude_particle_reservoir:
            reservoir_particles_indexes = np.where(z < 0)
            z = np.delete(z, reservoir_particles_indexes)
            for k in payload.keys():
                payload[k] = np.delete(payload[k], reservoir_particles_indexes)

        x = np.full_like(z, 0)
        y = np.full_like(z, 0)

        pointsToVTK(path, x, y, z, data=payload)

    def _add_leading_zeros(self, a):
        return "".join(["0" for i in range(self.num_len - len(str(a)))]) + str(a)

    def run(self):
        for time_index_and_value in enumerate(self.settings.save_spec_and_attr_times):
            self._export_attributes(time_index_and_value)


def readVTK_1D(file):

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file)
    reader.Update()

    vtk_output = reader.GetOutput()

    z = np.zeros(vtk_output.GetNumberOfPoints())
    for i in range(vtk_output.GetNumberOfPoints()):
        x, y, z[i] = vtk_output.GetPoint(i)

    data = {}
    data["z"] = z
    for i in range(vtk_output.GetPointData().GetNumberOfArrays()):
        data[vtk_output.GetPointData().GetArrayName(i)] = VN.vtk_to_numpy(
            vtk_output.GetPointData().GetArray(i)
        )

    return data
