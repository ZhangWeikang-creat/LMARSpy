import datetime

from lmarspy.backend.field import Field
from lmarspy.backend.cal import cal
from lmarspy.model.mp import Mp
from lmarspy.model.duogrid import Duogrid
from lmarspy.model.flag import Flag
from lmarspy.fps.data import FpsData

class AtmData:

    def __init__(self, fps: FpsData, fg: Flag):

        self.fps = fps

        self.fg = fg

        #--- setting times
        YYYY = fg.YYYY
        MM = fg.MM
        DD = fg.DD

        days = fg.days
        hours = fg.hours
        minutes = fg.minutes
        seconds = fg.seconds

        #--- dt
        dt_atmos = fg.dt_atmos 

        #--- dimensions
        npx = fps.npx
        npy = fps.npy

        npz = fg.npz
        ng = fg.ng

        fps.main_print('    Initializing communications...')

        mp = Mp(npx, npy, npz, ng)
        self.mp = mp

        fps.main_print('    Initializing computational grids...')

        self.dg = Duogrid(mp)
        self.dg.assign_cart_metrics()

        fps.main_print('    Initializing Atm vars...')

        #--- progs
        self.delp = Field(dims = mp.dim_va)
        self.delz = Field(dims = mp.dim_va)
        self.ua = Field(dims = mp.dim_va)
        self.va = Field(dims = mp.dim_va)
        self.wa = Field(dims = mp.dim_va)
        self.pt = Field(dims = mp.dim_va)

        #--- aux
        self.ps = Field(dims = mp.dim_ha)
        self.phis = Field(dims = mp.dim_ha)

        self.ptop = Field(data=0.)

        fps.main_print('    Initializing time...')

        self.time_init = datetime.datetime(YYYY,MM,DD)
        self.time_atmos = self.time_init
        sdt = datetime.timedelta(seconds = 86400*days + 3600*hours + 60*minutes + seconds)
        self.time_end = self.time_init + sdt
        self.num_atmos_calls = round(sdt.seconds/dt_atmos)
        self.dt_atmos = dt_atmos

        self.times = []
        dt = datetime.timedelta(seconds=dt_atmos)

        out_fre = fg.out_fre

        for i in range(0, self.num_atmos_calls+1, out_fre):
            self.times.append(self.time_init + i*dt)

        #--- check errors
        if self.num_atmos_calls//out_fre != self.num_atmos_calls/out_fre:
            raise ValueError('num_atmos_calls/out_fre must be int, and there will be no result for the last time step!')
        
        if not self.num_atmos_calls*dt_atmos == sdt.seconds:
            raise ValueError('Run length must be multiple of atmosphere time step')

        return



