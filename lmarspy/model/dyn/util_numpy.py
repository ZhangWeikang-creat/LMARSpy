from lmarspy.backend.field import Field
from lmarspy.backend.cal import cal
from lmarspy.model.const import Const

from lmarspy.model.mp import Mp
from lmarspy.model.duogrid import Duogrid

from .reconst_numpy import DynRc

import numpy as np
from numba import jit

class DynUtil:

    @staticmethod
    @jit(nopython=True)
    def get_x_upwind(pc, pw, pe, ut):

        pc[:] = ( ut > 0. )*pe + ( ut < 0. )*pw

        return
    
    @staticmethod
    @jit(nopython=True)
    def get_y_upwind(pc, ps, pn, vt):

        pc[:] = ( vt > 0. )*pn + ( vt < 0. )*ps

        return


    @staticmethod
    @jit(nopython=True)
    def get_z_upwind(pc, pto, pdn, wt):

        pc[1:-1,:,:] = ( wt[1:-1,:,:] > 0. )*pto[1:,:,:] + \
                ( wt[1:-1,:,:] < 0. )*pdn[:-1,:,:]

        return


    @staticmethod
    @jit(nopython=True)
    def get_lmars_x(ut, prc, ppc, uw, ue, prw, pre, cs, dp, dz, wk, ib: int, ie: int, jb: int, je: int):

        GRAV = 9.80665

        wk[:,:,ib:ie+1] = -0.5*(cs[:,:,ib-1:ie] + cs[:,:,ib:ie+1])* \
                (dp[:,:,ib-1:ie] + dp[:,:,ib:ie+1]) / GRAV / \
                (dz[:,:,ib-1:ie] + dz[:,:,ib:ie+1])

        ppc[:] = 0.5*wk[:,jb:je,ib:ie+1]*(ue - uw)
        prc[:] = 0.5*(pre + prw) + ppc

        ut[:] = 0.5*(ue + uw) + 0.5/wk[:,jb:je,ib:ie+1]*(pre - prw)

        return
    
    @staticmethod
    @jit(nopython=True)
    def get_lmars_y(vt, prc, ppc, vs, vn, prs, prn, cs, dp, dz, wk, ib: int, ie: int, jb: int, je: int):

        GRAV = 9.80665

        wk[:,jb:je+1,:] = -0.5*(cs[:,jb-1:je,:] + cs[:,jb:je+1,:])* \
                (dp[:,jb-1:je,:] + dp[:,jb:je+1,:]) / GRAV / \
                (dz[:,jb-1:je,:] + dz[:,jb:je+1,:])

        ppc[:] = 0.5*wk[:,jb:je+1,ib:ie]*(vn - vs)
        prc[:] = 0.5*(prn + prs) + ppc

        vt[:] = 0.5*(vn + vs) + 0.5/wk[:,jb:je+1,ib:ie]*(prn - prs)

        return


    @staticmethod
    @jit(nopython=True)
    def get_lmars_z(wt, pae, ppe, wto, wdn, pato, padn, cs, dp, dz, wke, ib: int, ie: int, jb: int, je: int):

        GRAV = 9.80665

        wke[1:-1,:,:] = -0.5*(cs[1:,:,:] + cs[:-1,:,:])* \
                (dp[1:,:,:] + dp[:-1,:,:]) / GRAV / \
                (dz[1:,:,:] + dz[:-1,:,:])

        wke[0,:,:] = -cs[0,:,:] * dp[0,:,:] / GRAV / dz[0,:,:]
        wke[-1,:,:] = -cs[-1,:,:] * dp[-1,:,:] / GRAV / dz[-1,:,:]


        ppe[1:-1,:,:] = 0.5*wke[1:-1,jb:je,ib:ie]*(wto[1:,:,:] - wdn[:-1,:,:])
        pae[1:-1,:,:] = 0.5*(pato[1:,:,:] + padn[:-1,:,:]) + ppe[1:-1,:,:]

        wt[1:-1,:,:] = 0.5*(wto[1:,:,:] + wdn[:-1,:,:])+ \
                0.5/wke[1:-1,jb:je,ib:ie]*(pato[1:,:,:] - padn[:-1,:,:])
        
        # 反射边界条件
        pae[-1,:,:] = padn[-1,:,:] - wke[-1,jb:je,ib:ie]*wdn[-1,:,:]
        wt[-1,:,:] = 0
        pae[0,:,:]  = pato[0,:,:] + wke[0,jb:je,ib:ie]*wto[0,:,:]
        wt[0,:,:]  = 0

        # 上边界开放，下边界反射
        #pae[-1,:,:] = padn[-1,:,:] - wke[-1,jb:je,ib:ie]*wdn[-1,:,:]
        #wt[-1,:,:] = 0
        #pae[0,:,:]  = pato[0,:,:]
        #wt[0,:,:]  = wto[0,:,:]
        

        return


    @staticmethod
    def get_crx(crx: Field, ut: Field, dt: Field, dg: Field, mp: Mp):
        # todo, assign rsina in dg

        ib  = mp.ib
        ie  = mp.ie

        rdxsina = cal.unsqueeze(dg.a_rdx / dg.a_sina, dim=0)
        rdxsinc = Field(dims=mp.dim_vc)

        DynUtil.get_x_upwind(rdxsinc, rdxsina, rdxsina, ut, ib, ie)

        crx[:] = ut * dt * rdxsinc

        return
    
    @staticmethod
    def get_cry(cry: Field, vt: Field, dt: Field, dg: Duogrid, mp: Mp):
        # todo, assign rsina in dg

        jb  = mp.jb
        je  = mp.je

        rdysina = cal.unsqueeze(dg.a_rdy / dg.a_sina, dim=0)
        rdysinc = Field(dims=mp.dim_vd)

        DynUtil.get_y_upwind(rdysinc, rdysina, rdysina, vt, jb, je)

        cry[:] = vt * dt * rdysinc

        return


    @staticmethod
    def get_crz(crz: Field, wt: Field, dt: Field, dz: Field, mp: Mp):
        # crz positive means upward

        rdza = -1./dz
        rdzc = Field(dims=mp.dim_vp)

        DynUtil.get_z_upwind(rdzc, rdza, rdza, wt)

        crz[:] = wt * dt * rdzc

        return
    
    @staticmethod
    @jit(nopython=True)
    def get_xfx(xfx, ut, dt: float, c_dy):

        xfx[:] = ut * c_dy * dt

        return
    
    @staticmethod
    @jit(nopython=True)
    def get_yfx(yfx, vt, dt:float, d_dx):

        yfx[:] = vt * d_dx * dt

        return


    @staticmethod
    @jit(nopython=True)
    def get_zfx(zfx, wt, dt: float):

        zfx[:] = wt * dt

        return

    @staticmethod
    @jit(nopython=True)
    def update_eul(rhs_qa, fx, fy, fz, dz, a_rda):

        rhs_qa[:] += ((fx[:,:,:-1] - fx[:,:,1:]) + (fy[:,:-1,:] - fy[:,1:,:]))\
                    * a_rda + (fz[:-1,:,:] - fz[1:,:,:]) / dz

        return
    
    @staticmethod
    @jit(nopython=True)
    def apply_pgrad_x_eul(rhs_ua, prc, dz, a_rdx, dt: float):

        grav = 9.80665

        rhs_ua[:] += -grav*dz * (prc[:,:,:-1] - prc[:,:,1:]) * a_rdx * dt

        return
    
    @staticmethod
    @jit(nopython=True)
    def apply_pgrad_y_eul(rhs_va, prc, dz, a_rdy, dt:float):

        grav = 9.80665

        rhs_va[:] += -grav*dz * (prc[:,:-1,:] - prc[:,1:,:]) * a_rdy * dt

        return


    @staticmethod
    @jit(nopython=True)
    def apply_pgrad_z_eul(rhs_wa, pae, dt: float):

        grav = 9.80665

        rhs_wa[:] += grav * (pae[1:,:,:] - pae[:-1,:,:]) * dt

        return
    
    @staticmethod
    @jit(nopython=True)
    def apply_grav_en_eul(rhs_en, dp, wa, dt: float):

        grav = 9.80665

        rhs_en[:] += -dp * grav * wa * dt

        return
    
    @staticmethod
    @jit(nopython=True)
    def apply_grav_wa_eul(rhs_wa, dp, dt: float):

        grav = 9.80665

        rhs_wa[:] += -dp * grav * dt

        return

    @staticmethod
    def get_var_we(pa: Field, pw: Field, pe: Field, mp: Mp, ng: int=1) -> None:

        ib = mp.ib
        ie = mp.ie
        jb = mp.jb
        je = mp.je

        if ng == 1:
            DynRc.get_var_we_ng1(pa[:,jb:je,ib-1:ie+1], pw, pe)

        elif ng == 2:
            DynRc.get_var_we_ng2(pa[:,jb:je,ib-2:ie+2], pw, pe)

        elif ng == 3:
            DynRc.get_var_we_ng3(pa[:,jb:je,ib-3:ie+3], pw, pe)

        else:
            raise ValueError(f'Unsupported ng value: {ng}')
        
        return
    
    @staticmethod
    def get_var_sn(pa: Field, ps: Field, pn: Field, mp: Mp, ng: int=1) -> None:

        ib = mp.ib
        ie = mp.ie
        jb = mp.jb
        je = mp.je

        if ng == 1:
            DynRc.get_var_sn_ng1(pa[:,jb-1:je+1,ib:ie], ps, pn)

        elif ng == 2:
            DynRc.get_var_sn_ng2(pa[:,jb-2:je+2,ib:ie], ps, pn)

        elif ng == 3:
            DynRc.get_var_sn_ng3(pa[:,jb-3:je+3,ib:ie], ps, pn)

        else:
            raise ValueError(f'Unsupported ng value: {ng}')
        
        return
    
    @staticmethod
    def get_var_td(pa: Field, pto: Field, pdn: Field, mp: Mp, ng: int=1) -> None:

        ib = mp.ib
        ie = mp.ie
        jb = mp.jb
        je = mp.je

        if ng == 1:
            DynRc.get_var_td_ng1(pa[:,jb:je,ib:ie], pto, pdn)

        elif ng == 2:
            DynRc.get_var_td_ng2(pa[:,jb:je,ib:ie], pto, pdn)

        elif ng == 3:
            DynRc.get_var_td_ng3(pa[:,jb:je,ib:ie], pto, pdn)

        else:
            raise ValueError(f'Unsupported ng value: {ng}')
        
        return
    
    @staticmethod
    def get_var_we_lim(pa: Field, pw: Field, pe: Field, mp: Mp, ng: int=1, lim_deg: float=1.) -> None:

        ib = mp.ib
        ie = mp.ie
        jb = mp.jb
        je = mp.je

        if ng == 1:
            DynRc.get_var_we_ng1(pa[:,jb:je,ib-1:ie+1], pw, pe)

        elif ng == 2:
            DynRc.get_var_we_lim_ng2(pa[:,jb:je,ib-2:ie+2], pw, pe, lim_deg)

        elif ng == 3:
            DynRc.get_var_we_lim_ng3(pa[:,jb:je,ib-3:ie+3], pw, pe, lim_deg)

        else:
            raise ValueError(f'Unsupported ng value: {ng}')
        
        return
    
    @staticmethod
    def get_var_sn_lim(pa: Field, ps: Field, pn: Field, mp: Mp, ng: int=1, lim_deg: float=1.) -> None:

        ib = mp.ib
        ie = mp.ie
        jb = mp.jb
        je = mp.je

        if ng == 1:
            DynRc.get_var_sn_ng1(pa[:,jb-1:je+1,ib:ie], ps, pn)

        elif ng == 2:
            DynRc.get_var_sn_lim_ng2(pa[:,jb-2:je+2,ib:ie], ps, pn, lim_deg)

        elif ng == 3:
            DynRc.get_var_sn_lim_ng3(pa[:,jb-3:je+3,ib:ie], ps, pn, lim_deg)

        else:
            raise ValueError(f'Unsupported ng value: {ng}')
        
        return
    
    @staticmethod
    def get_var_td_lim(pa: Field, pto: Field, pdn: Field, mp: Mp, ng: int=1, lim_deg: float=1.) -> None:

        ib = mp.ib
        ie = mp.ie
        jb = mp.jb
        je = mp.je

        if ng == 1:
            DynRc.get_var_td_ng1(pa[:,jb:je,ib:ie], pto, pdn)

        elif ng == 2:
            DynRc.get_var_td_lim_ng2(pa[:,jb:je,ib:ie], pto, pdn, lim_deg)

        elif ng == 3:
            DynRc.get_var_td_lim_ng3(pa[:,jb:je,ib:ie], pto, pdn, lim_deg)

        else:
            raise ValueError(f'Unsupported ng value: {ng}')
        
        return
