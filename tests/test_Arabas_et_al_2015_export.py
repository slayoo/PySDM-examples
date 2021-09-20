from PySDM.exporters import NetCDFExporter, VTKExporter
from PySDM_examples.Szumowski_et_al_1998.storage import Storage
from PySDM_examples.Szumowski_et_al_1998.gui_settings import GUISettings
from PySDM_examples.Szumowski_et_al_1998.simulation import Simulation
from PySDM_examples.Arabas_et_al_2015 import Settings, SpinUp
from PySDM_examples.utils import DummyController
from atmos_cloud_sim_uj_utils import TemporaryFile
from PySDM_examples.utils.widgets import IntSlider
from PySDM.backends import CPU
from tempfile import TemporaryDirectory
from scipy.io import netcdf


def test_Arabas_et_al_2015_export():
    # Arrange
    settings = GUISettings(Settings())
    settings.ui_nz.value += 1
    settings.ui_simulation_time = IntSlider(value=10)
    settings.ui_dt = IntSlider(value=10)
    settings.ui_output_options['interval'] = IntSlider(value=settings.ui_dt.value)
    assert settings.n_steps == 1
    assert len(settings.output_steps) == 2 and settings.output_steps[-1] == 1

    storage = Storage()
    simulator = Simulation(settings=settings, storage=storage, SpinUp=SpinUp, backend=CPU)
    file = TemporaryFile()
    ncdf_exporter = NetCDFExporter(storage=storage, settings=settings, simulator=simulator, filename=file.absolute_path)
    tempdir = TemporaryDirectory()
    vtk_exporter = VTKExporter(path=tempdir.name)

    # Act
    simulator.reinit()
    simulator.run(vtk_exporter=vtk_exporter)
    ncdf_exporter.run(controller=DummyController())

    # Assert
    versions = netcdf.netcdf_file(file.absolute_path).versions
    assert 'PyMPDATA' in str(versions)
