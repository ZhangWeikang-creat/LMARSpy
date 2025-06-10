# Purpose: controller of the LMARS dyn core


from lmarspy.model.sci import Sci

from .data import DynData
from .lag import DynLag
from lmarspy.model.const import Const

class DynCtrl:

    @staticmethod
    def step(dyn: DynData) -> None:
        '''stepping at dt_atmos level'''

        #--- get cs
        if dyn.sw_dyn == True:
            dyn.cs[:] = Sci.get_cs_sw(dyn.dp)
        else:
            dyn.cs[:] = Sci.get_cs(dyn.ptop, dyn.dp, dyn.dz)

        #--- to get value from fg 
        k_split = dyn.k_split

        #--- get base and medium time step
        bdt = dyn.dt_atmos
        mdt = bdt/k_split

        fg = dyn.fg

        #--- convert: T to PT (dyn)
        if dyn.pt_is_T and not dyn.sw_dyn:
            dyn.pt[:] = Sci.t2pt_dyn(dyn.pt, dyn.dp, dyn.dz)
            dyn.pt_is_T = False

        fg = dyn.fg
        #--- get rho en
        dyn.rho[:] = -dyn.dp/dyn.dz/Const.GRAV
        if fg.dyn_core == "eul_en_vic" or fg.dyn_core == "eul_en" or fg.dyn_core == "eul_en_dis":
            Sci.get_en_from_pt(dyn.en, dyn.rho, dyn.ua, dyn.va, dyn.wa, dyn.pt)

        #--- k_split - remap-time-level step
        for n_map in range(1, k_split+1):

            DynLag.step(dyn, mdt)

        if fg.dyn_core == "eul_en_vic" or fg.dyn_core == "eul_en" or fg.dyn_core == "eul_en_dis":
            #--- get pt
            dyn.rho[:] = -dyn.dp/dyn.dz/Const.GRAV
            Sci.get_pt_from_en(dyn.pt, dyn.rho, dyn.ua, dyn.va, dyn.wa, dyn.en)

        #--- recover: PT to T (dyn)
        if not dyn.pt_is_T and  not dyn.sw_dyn:
            dyn.pt[:] = Sci.pt2t_dyn(dyn.pt, dyn.dp, dyn.dz)
            dyn.pt_is_T = True


        return


