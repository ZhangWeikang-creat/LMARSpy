from mpi4py import MPI

from .data import FpsData
from .base import FpsBase
from lmarspy.backend.field import Field
from lmarspy.backend.cal import cal

class FpsDispatch:

    @staticmethod
    def gather_sum(fps: FpsData, local_data):
         
        if Field.backend == "torch":
            global_data = fps.comm.reduce(local_data.cpu(), op=MPI.SUM, root=fps.main_rank)

        elif Field.backend == 'numpy':
            global_data = fps.comm.reduce(local_data, op=MPI.SUM, root=fps.main_rank)

        else:
            FpsBase.raise_error(fps, f'Unsurpported backend: {Field.backend}')

        global_data = fps.comm.bcast(global_data, root=fps.main_rank)

        return Field(data=global_data)

    @staticmethod
    def gather_min(fps: FpsData, local_data):
         
        if Field.backend == "torch":
            global_data = fps.comm.reduce(local_data.cpu(), op=MPI.MIN, root=fps.main_rank)

        elif Field.backend == 'numpy':
            global_data = fps.comm.reduce(local_data, op=MPI.MIN, root=fps.main_rank)

        else:
            FpsBase.raise_error(fps, f'Unsurpported backend: {Field.backend}')

        global_data = fps.comm.bcast(global_data, root=fps.main_rank)

        return Field(data=global_data)
    
    @staticmethod
    def gather_max(fps: FpsData, local_data):
         
        if Field.backend == "torch":
            global_data = fps.comm.reduce(local_data.cpu(), op=MPI.MAX, root=fps.main_rank)

        elif Field.backend == 'numpy':
            global_data = fps.comm.reduce(local_data, op=MPI.MAX, root=fps.main_rank)

        else:
            FpsBase.raise_error(fps, f'Unsurpported backend: {Field.backend}')

        global_data = fps.comm.bcast(global_data, root=fps.main_rank)

        return Field(data=global_data)