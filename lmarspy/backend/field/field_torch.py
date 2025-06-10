import torch
from typing import List


class FieldTorch:

    def __new__(cls, data=None, dims: List[int]=None, device: str="cpu", dty: str='float64'):

        
        if device == "gpu":
            import os
            device_num = os.environ['LOCAL_RANK']
            dev = torch.device(f'cuda:{device_num}')
        elif device == "cpu":
            dev = device
        else:
            raise ValueError(f'Unsurpported device: {device} in torch')

        if dty == "float64":
            dtype = torch.float64
        elif dty == "float32":
            dtype = torch.float32
        else:
            raise ValueError(f'Unsupported dtype {dty}')


        if data is not None:
            if isinstance(data, torch.Tensor):
                '''How about no_grad?'''
                return data.detach().clone().type(dtype).to(dev)
            else:
                return torch.tensor(data,
                                    dtype=dtype,
                                    device=dev,
                                    requires_grad=False,
                                    )

        if dims is not None:
            return torch.zeros(dims,
                               dtype=dtype,
                               device=dev,
                               requires_grad=False,
                               )

        raise ValueError('data and dims cannot both be None.')
