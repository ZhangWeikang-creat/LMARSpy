# Purpose: Atm IO module

from mpi4py import MPI
import netCDF4 as nc
import numpy as np
import os

from lmarspy.backend.field import Field
from lmarspy.backend.cal import cal
from lmarspy.model.sci import Sci
from .data import AtmData

class AtmIO:

    @staticmethod
    def create_file_global(atm: AtmData) -> None:

        fps = atm.fps
        #--- define dataset
        ds = nc.Dataset(atm.nc_file, mode="w")

        if Field.backend == 'torch':
            x = atm.global_x.cpu()
            y = atm.global_y.cpu()
            z = atm.z.cpu()
        elif Field.backend == 'numpy':
            x = atm.global_x
            y = atm.global_y
            z = atm.z
        else:
            raise ValueError(f"Unsurpported backend: {Field.backend}")

        times = atm.times

        #--- define dimensions except time
        dtype = np.float32
        dim_table = (
            # name, dtype,    value
            ('z',   dtype, z),
            ('y',   dtype, y),
            ('x',   dtype, x),
        )
        for name, dtype, value in dim_table:
            if name not in ds.dimensions:
                ds.createDimension(name, len(value))
                var = ds.createVariable(name, dtype, (name,))
                var[:] = value

        #--- define time dimension
        ds.createDimension('time', None)
        var_time = ds.createVariable('time', np.float64, ('time',))
        var_time.units = f'seconds since {atm.time_init.strftime("%Y-%m-%d %H:%M:%S")}'
        var_time.calendar = 'proleptic_gregorian'

        for tid in range(len(times)):
            var_time[tid:] = nc.date2num([times[tid]], units=var_time.units, calendar=var_time.calendar)

        #--- define vars
        var_table = ('pt', 'dz', 'ua', 'va', 'wa', 'pot', 'dp')
        for var_name in var_table:
            ds.createVariable(var_name, dtype, ('time', 'z', 'y', 'x'))

        ds.close()

        return

    @staticmethod
    def create_file_global_parallel(atm: AtmData) -> None:#--- parallel会导致不同数量进程运行的文件的md5码不一致

        fps = atm.fps
        #--- define dataset
        ds = nc.Dataset(atm.nc_file, mode="w", parallel=True, comm=fps.comm, info=MPI.Info())

        if Field.backend == 'torch':
            x = atm.global_x.cpu()
            y = atm.global_y.cpu()
            z = atm.z.cpu()
        elif Field.backend == 'numpy':
            x = atm.global_x
            y = atm.global_y
            z = atm.z
        else:
            raise ValueError(f"Unsurpported backend: {Field.backend}")

        times = atm.times

        #--- define dimensions except time
        dtype = np.float32
        dim_table = (
            # name, dtype,    value
            ('z',   dtype, z),
            ('y',   dtype, y),
            ('x',   dtype, x),
        )
        for name, dtype, value in dim_table:
            if name not in ds.dimensions:
                ds.createDimension(name, len(value))
                var = ds.createVariable(name, dtype, (name,))
                var[:] = value

        #--- define time dimension
        ds.createDimension('time', None)
        var_time = ds.createVariable('time', np.float64, ('time',))
        var_time.units = f'seconds since {atm.time_init.strftime("%Y-%m-%d %H:%M:%S")}'
        var_time.calendar = 'proleptic_gregorian'

        var_time.set_collective(True)
        for tid in range(len(times)):
            var_time[tid:] = nc.date2num([times[tid]], units=var_time.units, calendar=var_time.calendar)

        #--- define vars
        var_table = ('pt', 'dz', 'ua', 'va', 'wa', 'pot', 'dp')
        for var_name in var_table:
            ds.createVariable(var_name, dtype, ('time', 'z', 'y', 'x'))

        ds.close()

        return

    @staticmethod
    def write_data_parallel(atm: AtmData, tid: int) -> None:

        fps = atm.fps
        #--- dims
        ibd = atm.mp.ibd
        ib  = atm.mp.ib
        ie  = atm.mp.ie
        ied = atm.mp.ied

        jbd = atm.mp.jbd
        jb  = atm.mp.jb
        je  = atm.mp.je
        jed = atm.mp.jed

        x_start = fps.x_start
        x_end  = fps.x_end
        y_start  = fps.y_start
        y_end = fps.y_end

        #--- grids, vars
        if Field.backend == 'torch':

            pt = atm.pt.cpu()[:,jb:je,ib:ie]
            dp = atm.delp.cpu()[:,jb:je,ib:ie]
            dz = atm.delz.cpu()[:,jb:je,ib:ie]
            ua = atm.ua.cpu()[:,jb:je,ib:ie]
            va = atm.va.cpu()[:,jb:je,ib:ie]
            wa = atm.wa.cpu()[:,jb:je,ib:ie]
            pot = (Sci.t2pt(atm.pt, atm.delp, atm.delz)).cpu()[:,jb:je,ib:ie]

        elif Field.backend == 'cupy':
            
            pt = atm.pt.get()[:,jb:je,ib:ie]
            dp = atm.delp.get()[:,jb:je,ib:ie]
            dz = atm.delz.get()[:,jb:je,ib:ie]
            ua = atm.ua.get()[:,jb:je,ib:ie]
            va = atm.va.get()[:,jb:je,ib:ie]
            wa = atm.wa.get()[:,jb:je,ib:ie]
            pot = (Sci.t2pt(atm.pt, atm.delp, atm.delz)).get()[:,jb:je,ib:ie]

        elif Field.backend == 'numpy':
            
            pt = atm.pt[:,jb:je,ib:ie]
            dp = atm.delp[:,jb:je,ib:ie]
            dz = atm.delz[:,jb:je,ib:ie]
            ua = atm.ua[:,jb:je,ib:ie]
            va = atm.va[:,jb:je,ib:ie]
            wa = atm.wa[:,jb:je,ib:ie]
            pot = (Sci.t2pt(atm.pt, atm.delp, atm.delz))[:,jb:je,ib:ie]

        else:
            raise ValueError(f"Unsurpported backend: {Field.backend}")

        #--- define dataset
        ds = nc.Dataset(atm.nc_file, mode="a", parallel=True, comm=fps.comm, info=MPI.Info())

        pt = np.expand_dims(pt, axis=0)
        dp = np.expand_dims(dp, axis=0)
        dz = np.expand_dims(dz, axis=0)
        pot = np.expand_dims(pot, axis=0)
        ua = np.expand_dims(ua, axis=0)
        va = np.expand_dims(va, axis=0)
        wa = np.expand_dims(wa, axis=0)

        var_table = (
            # name, value
            ('pt',  pt),
            ('dp',  dp),
            ('dz',  dz),
            ('pot', pot),
            ('ua',  ua),
            ('va',  va),
            ('wa',  wa),
        )
        for name, value in var_table:
            var = ds.variables[name]
            var.set_collective(True)
            var[tid, :, y_start:y_end, x_start:x_end] = value

        ds.close()

        return

    @staticmethod
    def create_file(atm: AtmData) -> None:

        fps = atm.fps
        #--- define dataset
        ds = nc.Dataset(atm.nc_file[:-3]+str(fps.rank+1)+'.nc', mode="w")

        if Field.backend == 'torch':
            x = atm.x.cpu()
            y = atm.y.cpu()
            z = atm.z.cpu()
        elif Field.backend == 'numpy':
            x = atm.x
            y = atm.y
            z = atm.z
        else:
            raise ValueError(f"Unsurpported backend: {Field.backend}")

        times = atm.times
        dtype = np.float64

        #--- define dimensions except time
        dim_table = (
            # name, dtype,    value
            ('z',   dtype, z),
            ('y',   dtype, y),
            ('x',   dtype, x),
        )
        for name, dtype, value in dim_table:
            if name not in ds.dimensions:
                ds.createDimension(name, len(value))
                var = ds.createVariable(name, dtype, (name,))
                var[:] = value

        #--- define time dimension
        ds.createDimension('time', None)
        var_time = ds.createVariable('time', np.float64, ('time',))
        var_time.units = f'seconds since {atm.time_init.strftime("%Y-%m-%d %H:%M:%S")}'
        var_time.calendar = 'proleptic_gregorian'

        for tid in range(len(times)):
            var_time[tid:] = nc.date2num([times[tid]], units=var_time.units, calendar=var_time.calendar)

        #--- define vars
        var_table = ('pt', 'dz', 'ua', 'va', 'wa', 'pot', 'dp')
        for var_name in var_table:
            ds.createVariable(var_name, dtype, ('time', 'z', 'y', 'x'))

        ds.close()

        return

    @staticmethod
    def write_data(atm: AtmData, tid: int) -> None:

        fps = atm.fps
        #--- dims
        ibd = atm.mp.ibd
        ib  = atm.mp.ib
        ie  = atm.mp.ie
        ied = atm.mp.ied

        jbd = atm.mp.jbd
        jb  = atm.mp.jb
        je  = atm.mp.je
        jed = atm.mp.jed

        x_start = fps.x_start
        x_end  = fps.x_end
        y_start  = fps.y_start
        y_end = fps.y_end

        #--- grids, vars
        if Field.backend == 'torch':

            pt = atm.pt.cpu()[:,jb:je,ib:ie]
            dp = atm.delp.cpu()[:,jb:je,ib:ie]
            dz = atm.delz.cpu()[:,jb:je,ib:ie]
            ua = atm.ua.cpu()[:,jb:je,ib:ie]
            va = atm.va.cpu()[:,jb:je,ib:ie]
            wa = atm.wa.cpu()[:,jb:je,ib:ie]
            pot = (Sci.t2pt(atm.pt, atm.delp, atm.delz)).cpu()[:,jb:je,ib:ie]

        elif Field.backend == 'cupy':
            
            pt = atm.pt.get()[:,jb:je,ib:ie]
            dp = atm.delp.get()[:,jb:je,ib:ie]
            dz = atm.delz.get()[:,jb:je,ib:ie]
            ua = atm.ua.get()[:,jb:je,ib:ie]
            va = atm.va.get()[:,jb:je,ib:ie]
            wa = atm.wa.get()[:,jb:je,ib:ie]
            pot = (Sci.t2pt(atm.pt, atm.delp, atm.delz)).get()[:,jb:je,ib:ie]

        elif Field.backend == 'numpy':
            
            pt = atm.pt[:,jb:je,ib:ie]
            dp = atm.delp[:,jb:je,ib:ie]
            dz = atm.delz[:,jb:je,ib:ie]
            ua = atm.ua[:,jb:je,ib:ie]
            va = atm.va[:,jb:je,ib:ie]
            wa = atm.wa[:,jb:je,ib:ie]
            pot = (Sci.t2pt(atm.pt, atm.delp, atm.delz))[:,jb:je,ib:ie]

        else:
            raise ValueError(f"Unsurpported backend: {Field.backend}")

        #--- define dataset
        ds = nc.Dataset(atm.nc_file[:-3]+str(fps.rank+1)+'.nc', mode="a")

        pt = np.expand_dims(pt, axis=0)
        dp = np.expand_dims(dp, axis=0)
        dz = np.expand_dims(dz, axis=0)
        pot = np.expand_dims(pot, axis=0)
        ua = np.expand_dims(ua, axis=0)
        va = np.expand_dims(va, axis=0)
        wa = np.expand_dims(wa, axis=0)

        var_table = (
            # name, value
            ('pt',  pt),
            ('dp',  dp),
            ('dz',  dz),
            ('pot', pot),
            ('ua',  ua),
            ('va',  va),
            ('wa',  wa),
        )
        for name, value in var_table:
            var = ds.variables[name]
            var[tid] = value

        ds.close()

        return
    
    @staticmethod
    def merge_ncfile(atm: AtmData) -> None:

        fps = atm.fps
        
        #--- define dataset
        ds_all = nc.Dataset(atm.nc_file, mode="a")

        for i in range(fps.size):
            x_start, x_end, y_start, y_end = fps.subdomains[i]

            ds = nc.Dataset(atm.nc_file[:-3]+str(i+1)+'.nc', mode="r")

            for name in ["dp", "dz", "pt", "pot", "ua", "va", "wa"]:

                var_all = ds_all.variables[name]
                var = ds.variables[name]

                var_all[:,:, y_start:y_end, x_start:x_end] = var[:]
            
            ds.close()
            
            #--- remove son ncfile
            os.remove(atm.nc_file[:-3]+str(i+1)+'.nc')

        return
