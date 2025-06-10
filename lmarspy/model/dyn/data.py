# Purpose: data container of LMARS dynamical core

from lmarspy.backend.field import Field
from lmarspy.model.atm import Atm

class DynData:

    def __init__(self, atm: Atm):

        self.pt_is_T = True
        self.fps = atm.fps

        # objs
        self.fg = atm.fg
        self.mp = atm.mp
        self.dg = atm.dg
        
        fg = atm.fg
        mp = atm.mp
        dg = atm.dg
        self.atm = atm
        
        #--- from Atm
        self.dp = atm.delp
        self.dz = atm.delz
        self.ua = atm.ua
        self.va = atm.va
        self.wa = atm.wa
        self.pt = atm.pt
        
        #--- dim parameters
        ng = mp.ng
        npx = mp.npx
        npy = mp.npy
        npz = mp.npz

        ibd = mp.ibd
        ib  = mp.ib
        ie  = mp.ie
        ied = mp.ied

        jbd = mp.jbd
        jb  = mp.jb
        je  = mp.je
        jed = mp.jed

        fps = atm.fps
        
        self.dt_atmos = atm.dt_atmos
        #--- sw_dyn flags
        if fg.dyn_core == "eul_sw" or fg.dyn_core == "eul_sw_linear":
            self.sw_dyn = True
        else:
            self.sw_dyn = False

        #--- rk flags
        self.rk = fg.rk

        self.lim_deg = fg.lim_deg

        #--- mpi vars
        # Attention!!!
        # wrong max_count will cause boundary error 
        self.count = 0
        if 0 < self.rk < 10:
            self.max_count = fg.n_split*fg.k_split*1
        if 10 < self.rk < 100:
            self.max_count = fg.n_split*fg.k_split*2
        if 100 < self.rk < 1000:
            self.max_count = fg.n_split*fg.k_split*3

        self.dp_reqs = []
        self.dz_reqs = []
        self.pt_reqs = []
        self.ua_reqs = []
        self.va_reqs = []
        self.wa_reqs = []
        self.en_reqs = []

        self.dp_w_recv = Field(dims=(npz, jed ,ng))
        self.dz_w_recv = Field(dims=(npz, jed ,ng))
        self.pt_w_recv = Field(dims=(npz, jed ,ng))
        self.ua_w_recv = Field(dims=(npz, jed ,ng))
        self.va_w_recv = Field(dims=(npz, jed ,ng))
        self.wa_w_recv = Field(dims=(npz, jed ,ng))
        self.en_w_recv = Field(dims=(npz, jed ,ng))

        self.dp_e_recv = Field(dims=(npz, jed ,ng))
        self.dz_e_recv = Field(dims=(npz, jed ,ng))
        self.pt_e_recv = Field(dims=(npz, jed ,ng))
        self.ua_e_recv = Field(dims=(npz, jed ,ng))
        self.va_e_recv = Field(dims=(npz, jed ,ng))
        self.wa_e_recv = Field(dims=(npz, jed ,ng))
        self.en_e_recv = Field(dims=(npz, jed ,ng))
        
        self.dp_s_recv = Field(dims=(npz, ng, ied))
        self.dz_s_recv = Field(dims=(npz, ng, ied))
        self.pt_s_recv = Field(dims=(npz, ng, ied))
        self.ua_s_recv = Field(dims=(npz, ng, ied))
        self.va_s_recv = Field(dims=(npz, ng, ied))
        self.wa_s_recv = Field(dims=(npz, ng, ied))
        self.en_s_recv = Field(dims=(npz, ng, ied))

        self.dp_n_recv = Field(dims=(npz, ng, ied))
        self.dz_n_recv = Field(dims=(npz, ng, ied))
        self.pt_n_recv = Field(dims=(npz, ng, ied))
        self.ua_n_recv = Field(dims=(npz, ng, ied))
        self.va_n_recv = Field(dims=(npz, ng, ied))
        self.wa_n_recv = Field(dims=(npz, ng, ied))
        self.en_n_recv = Field(dims=(npz, ng, ied))

        # time para
        self.k_split = fg.k_split
        self.n_split = fg.n_split

        dtype = "float64"
        
        #--- work arrays
        self.ua0  = Field(dims = mp.dim_va)
        self.va0  = Field(dims = mp.dim_va)
        self.wa0  = Field(dims = mp.dim_va)
        self.dp0 = Field(dims = mp.dim_va)
        self.pt0 = Field(dims = mp.dim_va)
        self.dz0 = Field(dims = mp.dim_va)
        
        self.pr = Field(dims = mp.dim_va)
        self.pa = Field(dims = mp.dim_va)
        self.ph = Field(dims = mp.dim_va)
        self.pe = Field(dims = mp.dim_vp)
        
        self.cs = Field(dims = mp.dim_va)
        self.wk  = Field(dims = mp.dim_va)
        self.wke = Field(dims = mp.dim_vp)
        
        self.rho = Field(dims = mp.dim_va)
        self.rho0 = Field(dims = mp.dim_va)
        self.en = Field(dims = mp.dim_va)
        self.en0 = Field(dims = mp.dim_va)

        dim_vam = [npz, je-jb, ie-ib]
        dim_vcm = [npz, je-jb, ie-ib+1]
        dim_vdm = [npz, je-jb+1, ie-ib]
        dim_vpm = [npz+1, je-jb, ie-ib]
        dim_vpm2 = [npz+2, je-jb, ie-ib]

        self.rhs_ua = Field(dims = dim_vam, dty=dtype)
        self.rhs_va = Field(dims = dim_vam, dty=dtype)
        self.rhs_wa = Field(dims = dim_vam, dty=dtype)
        self.rhs_dp = Field(dims = dim_vam, dty=dtype)
        self.rhs_pt = Field(dims = dim_vam, dty=dtype)
        self.rhs_dz = Field(dims = dim_vam, dty=dtype)
        self.rhs_en = Field(dims = dim_vam, dty=dtype)

        self.fx = Field(dims = dim_vcm, dty=dtype)
        self.fy = Field(dims = dim_vdm, dty=dtype)
        self.fz = Field(dims = dim_vpm, dty=dtype)

        self.fxw = Field(dims = dim_vcm, dty=dtype)
        self.fxe = Field(dims = dim_vcm, dty=dtype)
        self.fys = Field(dims = dim_vdm, dty=dtype)
        self.fyn = Field(dims = dim_vdm, dty=dtype)
        self.fzto = Field(dims = dim_vam, dty=dtype)
        self.fzdn = Field(dims = dim_vam, dty=dtype)

        self.fxtr = Field(dims = dim_vcm, dty=dtype)
        self.fytr = Field(dims = dim_vdm, dty=dtype)
        self.fztr = Field(dims = dim_vpm, dty=dtype)

        self.fxtrw = Field(dims = dim_vcm, dty=dtype)
        self.fxtre = Field(dims = dim_vcm, dty=dtype)
        self.fytrs = Field(dims = dim_vdm, dty=dtype)
        self.fytrn = Field(dims = dim_vdm, dty=dtype)
        self.fztrto = Field(dims = dim_vam, dty=dtype)
        self.fztrdn = Field(dims = dim_vam, dty=dtype)

        self.uaw = Field(dims = dim_vcm, dty=dtype)
        self.uae = Field(dims = dim_vcm, dty=dtype)
        self.vas = Field(dims = dim_vdm, dty=dtype)
        self.van = Field(dims = dim_vdm, dty=dtype)
        self.wato = Field(dims = dim_vam, dty=dtype)
        self.wadn = Field(dims = dim_vam, dty=dtype)

        self.prw = Field(dims = dim_vcm, dty=dtype)
        self.pre = Field(dims = dim_vcm, dty=dtype)
        self.prs = Field(dims = dim_vdm, dty=dtype)
        self.prn = Field(dims = dim_vdm, dty=dtype)

        self.pato = Field(dims = dim_vam, dty=dtype)
        self.padn = Field(dims = dim_vam, dty=dtype)

        self.ut  = Field(dims = dim_vcm, dty=dtype)
        self.vt  = Field(dims = dim_vdm, dty=dtype)
        self.wt  = Field(dims = dim_vpm, dty=dtype)

        self.prcx = Field(dims = dim_vcm, dty=dtype)
        self.ppcx = Field(dims = dim_vcm, dty=dtype)
        self.prcy = Field(dims = dim_vdm, dty=dtype)
        self.ppcy = Field(dims = dim_vdm, dty=dtype)

        self.xfx = Field(dims = dim_vcm, dty=dtype)
        self.crx = Field(dims = dim_vcm, dty=dtype)
        self.yfx = Field(dims = dim_vdm, dty=dtype)
        self.cry = Field(dims = dim_vdm, dty=dtype)
        self.crz = Field(dims = dim_vpm, dty=dtype)
        self.zfx = Field(dims = dim_vpm, dty=dtype)

        #--- need when vic
        if fg.dyn_core == "eul_vic" or fg.dyn_core == "eul_en_vic":
            self.ua2 = Field(dims = mp.dim_va, dty=dtype)
            self.va2 = Field(dims = mp.dim_va, dty=dtype)
            self.wa2 = Field(dims = mp.dim_va, dty=dtype)
            self.dp2 = Field(dims = mp.dim_va, dty=dtype)
            self.pt2 = Field(dims = mp.dim_va, dty=dtype)
            self.dz2 = Field(dims = mp.dim_va, dty=dtype)
            self.rho2 = Field(dims = mp.dim_va, dty=dtype)
            self.en2 = Field(dims = mp.dim_va, dty=dtype)

            self.res = Field(dims = dim_vam + [5,1], dty=dtype)
            self.delta_R = Field(dims = dim_vam + [5,1], dty=dtype)
            
            self.jr = Field(dims = (dim_vpm + [5, 5]), dty=dtype)
            dim_vpm2 = dim_vam[:]
            dim_vpm2[0] += 2
            self.ja = Field(dims = (dim_vpm2 + [5, 5]), dty=dtype)

            self.ur = Field(dims = dim_vpm2, dty=dtype)
            self.vr = Field(dims = dim_vpm2, dty=dtype)
            self.wr = Field(dims = dim_vpm2, dty=dtype)
            self.hr = Field(dims = dim_vpm2, dty=dtype)
            self.rhor = Field(dims = dim_vpm2, dty=dtype)
            self.prr = Field(dims = dim_vpm2, dty=dtype)
            self.ptr = Field(dims = dim_vpm2, dty=dtype)

            self.rhoex = Field(dims = dim_vpm2, dty=dtype)
            self.uex = Field(dims = dim_vpm2, dty=dtype)
            self.vex = Field(dims = dim_vpm2, dty=dtype) 
            self.wex = Field(dims = dim_vpm2, dty=dtype)
            self.prex = Field(dims = dim_vpm2, dty=dtype)
            self.ptex = Field(dims = dim_vpm2, dty=dtype)
        
            if fg.dyn_core == "eul_en_vic":
                self.pp = Field(dims = (dim_vpm + [5, 5]), dty=dtype)
                self.ee = Field(dims = (dim_vpm + [5, 5]), dty=dtype)
                self.ppinv = Field(dims = (dim_vpm + [5, 5]), dty=dtype)
        
        #--- misc
        self.ptop = atm.ptop

        #--- init static fields
        self.pe[0,:,:]  = atm.ptop
        #self.gz = Field(dims = mp.dim_vpm)
        #self.gz[-1,:,:] = atm.phis

        self.pae = Field(dims = dim_vpm)
        self.ppe = Field(dims = dim_vpm)

        self.counter = 0

        return


