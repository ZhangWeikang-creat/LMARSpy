# Purpose: remap time-level LMARS dyn core controller

from .data import DynData
from .core import DynCore

class DynLag:

    @staticmethod
    def step(dyn: DynData, mdt: float) -> None:

        fps = dyn.fps

        mp = dyn.mp
        
        ua = dyn.ua
        va = dyn.va
        wa = dyn.wa
        dp = dyn.dp
        pt = dyn.pt
        dz = dyn.dz
        en = dyn.en
        rho = dyn.rho

        fg = dyn.fg

        #--- get n_split and dt
        n_split = dyn.n_split
        dt = mdt /n_split
        dt2 = 0.5*dt
        dt3 = dt/3
        rdt = 1/dt

        #--- main loop
        for it in range(n_split):

            #--- var0
            dyn.ua0[:]  = dyn.ua
            dyn.va0[:]  = dyn.va
            dyn.wa0[:]  = dyn.wa
            dyn.dp0[:]  = dyn.dp
            dyn.pt0[:]  = dyn.pt
            
            dyn.dz0[:]  = dyn.dz
            dyn.en0[:]  = dyn.en
            dyn.rho0[:]  = dyn.rho

            #--- acoustic level full subcycle(s), to do rk
            if dyn.rk == 333:
                DynCore.step(dyn, dt3, ng=3)
                DynCore.step(dyn, dt2, ng=3)
                DynCore.step(dyn, dt,  ng=3, lim=fg.lim)

            elif dyn.rk == 222:
                DynCore.step(dyn, dt3, ng=2)
                DynCore.step(dyn, dt2, ng=2)
                DynCore.step(dyn, dt,  ng=2, lim=fg.lim)

            elif dyn.rk == 111:
                DynCore.step(dyn, dt3, ng=1)
                DynCore.step(dyn, dt2, ng=1)
                DynCore.step(dyn, dt,  ng=1, lim=fg.lim)

            elif dyn.rk == 123:
                DynCore.step(dyn, dt3, ng=1)
                DynCore.step(dyn, dt2, ng=2)
                DynCore.step(dyn, dt,  ng=3, lim=fg.lim)
                
            elif dyn.rk == 13:
                DynCore.step(dyn, dt2, ng=1)
                DynCore.step(dyn, dt,  ng=3, lim=fg.lim)

            else:
                fps.raise_error(f"unsupported rk: {dyn.rk}")

        #--- remap (in the future)

        return



