import os
import tempfile
from pathlib import Path
import numpy as np


class Storage:
    class Exception(BaseException):
        pass

    def __init__(self, dtype=np.float32, path=None):
        if path is None:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.dir_path = self.temp_dir.name
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.dir_path = Path(path).absolute()
        self.dtype = dtype
        self.grid = None
        self._data_range = None

    def __del__(self):
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()

    def init(self, settings):
        self.grid = settings.grid
        self._data_range = {}

    def _filepath(self, name: str, step: int = None):
        if step is None:
            filename = f"{name}.npy"
        else:
            filename = f"{name}_{step:06}.npy"
        path = os.path.join(self.dir_path, filename)
        return path

    def save(self, data: (float, np.ndarray), step: int, name: str):
        if isinstance(data, (int, float)):
            path = self._filepath(name)
            np.save(path, np.concatenate((() if step == 0 else np.load(path), (self.dtype(data),))))
        elif data.shape[0:2] == self.grid:
            np.save(self._filepath(name, step), data.astype(self.dtype))
        else:
            raise NotImplementedError()

        if name not in self._data_range:
            self._data_range[name] = (np.inf, -np.inf)
        self._data_range[name] = (
            min(np.amin(data), self._data_range[name][0]),
            max(np.amax(data), self._data_range[name][1])
        )

    def data_range(self, name):
        return self._data_range[name]

    def load(self, name: str, step: int = None) -> np.ndarray:
        try:
            data = np.load(self._filepath(name, step))
        except FileNotFoundError as err:
            raise Storage.Exception() from err
        return data
