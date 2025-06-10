import numpy as np
from lmarspy.model.mp import Mp
from lmarspy.fps.data import FpsData

class ICdata:

    def __init__(self, fps: FpsData, mp: Mp):
        #--- dims
        self.npx = mp.npx
        self.npy = mp.npy

        self.global_npx = fps.global_npx
        self.global_npy = fps.global_npy

        self.npz = mp.npz
        self.ng  = mp.ng

        self.ibd = mp.ibd
        self.ib  = mp.ib
        self.ie  = mp.ie
        self.ied = mp.ied

        self.jbd = mp.jbd
        self.jb  = mp.jb
        self.je  = mp.je
        self.jed = mp.jed

        self.global_ibd = fps.global_ibd
        self.global_ib = fps.global_ib 
        self.global_ie = fps.global_ie
        self.global_ied = fps.global_ied

        self.global_jbd = fps.global_jbd
        self.global_jb = fps.global_jb
        self.global_je = fps.global_je
        self.global_jed = fps.global_jed

        #--- progs
        dtype = 'float64'
        delp = np.zeros(mp.dim_va, dtype=dtype)
        delz = np.zeros(mp.dim_va, dtype=dtype)
        ua = np.zeros(mp.dim_va, dtype=dtype)
        va = np.zeros(mp.dim_va, dtype=dtype)
        wa = np.zeros(mp.dim_va, dtype=dtype)
        pt = np.zeros(mp.dim_va, dtype=dtype)
        

        #--- aux
        ps = np.zeros(mp.dim_ha, dtype=dtype)
        phis = np.zeros(mp.dim_ha, dtype=dtype)

        ptop = 0.

        pk = np.zeros(mp.dim_vp, dtype=dtype)
        dpk = np.zeros(mp.dim_va, dtype=dtype)

        self.delp = delp
        self.delz = delz
        self.ua = ua
        self.va = va
        self.wa = wa
        self.pt = pt

        self.ps = ps
        self.phis = phis
        self.ptop = ptop

        self.pk = pk
        self.dpk = dpk

        return
    
            