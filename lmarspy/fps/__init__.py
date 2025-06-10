#!/usr/bin/env python

# Purpose: Flexible Parallel Computing Library
# Author: Chen Research Group (xichen.me@outlook.com)

from .data import FpsData
from .base import FpsBase
from .domain import FpsDomain
from .dispatch import FpsDispatch
from lmarspy.backend.field import Field
if Field.backend == "numpy":
    from .boundary_numpy import FpsBoundary
elif Field.backend == "torch":
    from .boundary_torch import FpsBoundary
else:
    raise ValueError(f'Unsurpported backend: {Field.backend}')

class Fps(FpsData):

    #--- base
    def print(self: FpsData, *args) -> None:

        FpsBase.print(self, args)

        return
    
    def main_print(self: FpsData, *args) -> None:

        FpsBase.main_print(self, args)

        return
    
    def raise_error(self: FpsData, info) -> None:

        FpsBase.raise_error(self, info)

        return
    
    def get_next_tag(self: FpsData):

        return FpsBase.get_next_tag(self)
    
    def init_model(self: FpsData):

        return FpsBase.init_model(self)
    
    def end_model(self: FpsData):

        return FpsBase.end_model(self)
    
    #--- domain
    def decompose_domain(self: FpsData, nx, ny, ng, px, py):
        
        return FpsDomain.decompose_domain(self, nx, ny, ng, px, py)
    
    #--- exchange boundary(未来转到duogrid)
    def exchange_boundary(self: FpsData, local_data):

        FpsBoundary.exchange_boundary(self, local_data)

    def exchange_boundary_create(self: FpsData, send_data, w_recv, e_recv, s_recv, n_recv, reqs):

        FpsBoundary.exchange_boundary_create(self, send_data, w_recv, e_recv, s_recv, n_recv, reqs)

    def exchange_boundary_wait(self: FpsData, send_data, w_recv, e_recv, s_recv, n_recv, reqs):

        FpsBoundary.exchange_boundary_wait(self, send_data, w_recv, e_recv, s_recv, n_recv, reqs)

    #--- dispatch
    def gather_sum(self: FpsData, local_data):

        return FpsDispatch.gather_sum(self, local_data)

    def gather_min(self: FpsData, local_data):

        return FpsDispatch.gather_min(self, local_data)

    def gather_max(self: FpsData, local_data):

        return FpsDispatch.gather_max(self, local_data)
    
    #--- diag
    def get_memory(self: FpsData):

        FpsBase.get_memory(self)



        