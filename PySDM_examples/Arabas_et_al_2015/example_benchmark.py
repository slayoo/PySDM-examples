from PySDM_examples.Arabas_et_al_2015 import Settings
from PySDM_examples.Szumowski_et_al_1998 import Simulation, Storage
from PySDM.products.stats.timers import WallTime
import PySDM.backends.numba.conf
from PySDM.backends import CPU, GPU
import importlib


def reload_cpu_backend():
    importlib.reload(PySDM.backends.numba.impl._algorithmic_methods)
    importlib.reload(PySDM.backends.numba.impl._index_methods)
    importlib.reload(PySDM.backends.numba.impl._pair_methods)
    importlib.reload(PySDM.backends.numba.impl._physics_methods)
    importlib.reload(PySDM.backends.numba.impl.storage_impl)
    importlib.reload(PySDM.backends)
    from PySDM.backends import CPU


def main():
    settings = Settings()

    settings.grid = (25, 25)
    settings.simulation_time = settings.dt * 100
    settings.output_interval = settings.dt * 10
    settings.processes = {
        "particle advection": True,
        "fluid advection": True,
        "coalescence": True,
        "condensation": False,
        "sedimentation": True,
        'freezing': False
    }

    n_sd = range(14, 16, 1)

    times = {}
    backends = [(CPU, "sync"), (CPU, "async")]
    if GPU.ENABLE:
        backends.append((GPU, "async"))
    for backend, mode in backends:
        if backend is CPU:
            PySDM.backends.numba.conf.NUMBA_PARALLEL = mode
            reload_cpu_backend()
        key = f"{backend} (mode={mode})"
        times[key] = []
        for sd in n_sd:
            settings.n_sd_per_gridbox = sd
            storage = Storage()
            simulation = Simulation(settings, storage, None, backend)
            simulation.reinit(products=[WallTime()])
            simulation.run()
            times[key].append(storage.load('wall_time')[-1])

    from matplotlib import pyplot as plt
    for parallelization, t in times.items():
        plt.plot(n_sd, t, label=parallelization)
    plt.legend()
    plt.loglog()
    plt.savefig("benchmark.pdf", format="pdf")


if __name__ == '__main__':
    main()
