#!/usr/bin/env python

# Purpose: General Scientific Computing Library field: numpy, torch, cupy
# Author: Chen Research Group (xichen.me@outlook.com)

from .field_numpy import FieldNP
from .field_torch import FieldTorch

import psutil
import platform

class Field:

    default_backend = 'numpy'
    backend = default_backend
    default_device = 'cpu'
    device = default_device

    def __new__(self,
                data=None,
                dims=None,
                dty="float64"
                ):
        if self.backend == 'numpy':
            return FieldNP(data=data, dims=dims, device=self.device, dty=dty)

        if self.backend == 'torch':
            return FieldTorch(data=data, dims=dims, device=self.device, dty=dty)

        raise ValueError(f'Unsurpported backend: {self.backend}')


    @classmethod
    def get_device(cls) -> None:

        if Field.backend == 'numpy':
            cpu_brand = platform.processor()

            return f"{cpu_brand}" + f"(物理核心数:{psutil.cpu_count(logical=False)}, 逻辑核心数:{psutil.cpu_count(logical=True)})"

        if Field.backend == 'torch':

            if Field.device == "gpu":

                import torch
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                return device_name + ":" + str(current_device)
            
            else:
                cpu_brand = platform.processor()
                
                return f"{cpu_brand}" + f"(物理核心数:{psutil.cpu_count(logical=False)}, 逻辑核心数:{psutil.cpu_count(logical=True)})"

    
        raise ValueError(f'Unsurpported backend: {Field.backend}')


