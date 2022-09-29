"""
deJong & Mackay SDM breakup paper figures
"""
# pylint: disable=invalid-name
from PySDM_examples.Arabas_et_al_2015.settings import Settings as Settings2D
from PySDM_examples.Shipway_and_Hill_2012.plot import plot as plot1D
from PySDM_examples.Shipway_and_Hill_2012.plot import plot_plusminus as plot1D_plusminus
from PySDM_examples.Shipway_and_Hill_2012.settings import Settings as Settings1D
from PySDM_examples.Shipway_and_Hill_2012.simulation import Simulation as Simulation1D
from PySDM_examples.Szumowski_et_al_1998.simulation import Simulation as Simulation2D

from .settings_0D import Settings0D
from .simulation_0D import run_box_breakup, run_box_NObreakup
