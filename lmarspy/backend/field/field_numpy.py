import numpy as np
from typing import List

class FieldNP:

    def __new__(cls, data=None, dims: List[int]=None, device: str="cpu", dty: str='float64'):

        if device != "cpu":
            raise ValueError(f'Unsurpported device: {device} in numpy')

        if dty == "float64":
            dtype = "float64"
        elif dty == "float32":
            dtype = "float32"
        else:
            raise ValueError(f'Unsupported dtype {dty}')

        if data is not None:
            return np.array(data, dtype=dtype)

        if dims is not None:
            return np.zeros(dims, dtype=dtype)

        raise ValueError('data and dims cannot both be None.')


