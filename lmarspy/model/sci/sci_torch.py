
import torch
from lmarspy.backend.field import Field
from lmarspy.backend.cal import cal

#--- Sci
class Sci:

    @staticmethod
    @torch.jit.script
    def pt2t(pt, dp, dz):
        '''Dry version for now'''

        GRAV = 9.80665
        RDGAS = 287.05
        CP_AIR = 1004.6
        KAPPA = RDGAS/CP_AIR

        rrg = - RDGAS / GRAV
        k1k = RDGAS / (CP_AIR - RDGAS)
        p00 = 1.e5
        rpk0 = 1/p00**KAPPA

        pt = pt * rpk0 * torch.exp(k1k*
                     torch.log( rrg * dp / dz * pt * rpk0 ))

        return pt


    @staticmethod
    def t2pt(pt, dp, dz):
        '''Dry version for now'''

        GRAV = 9.80665
        RDGAS = 287.05
        CP_AIR = 1004.6
        KAPPA = RDGAS/CP_AIR

        p00 = 1.e5
        pk0 = p00**KAPPA

        pt = Sci.t2pt_dyn(pt, dp, dz) * pk0

        return pt


    @staticmethod
    @torch.jit.script
    def pt2t_dyn(pt, dp, dz):
        '''Dry version for now'''

        GRAV = 9.80665
        RDGAS = 287.05
        CP_AIR = 1004.6
        KAPPA = RDGAS/CP_AIR

        rrg = - RDGAS / GRAV
        k1k = RDGAS / (CP_AIR - RDGAS)

        pt = pt * torch.exp(k1k* torch.log( rrg * dp / dz * pt ))

        return pt


    @staticmethod
    @torch.jit.script
    def t2pt_dyn(pt, dp, dz):
        '''Dry version for now'''

        GRAV = 9.80665
        RDGAS = 287.05
        CP_AIR = 1004.6
        KAPPA = RDGAS/CP_AIR
        rrg = - RDGAS / GRAV
        kappa = KAPPA

        pt = pt / torch.exp(kappa*torch.log( rrg * dp / dz * pt ))

        return pt


    @staticmethod
    def get_cs(ptop, dp, dz):
        '''need to be efficient, fast, no accuracy requirement'''

        GRAV = 9.80665
        RDGAS = 287.05
        CP_AIR = 1004.6
        KAPPA = RDGAS/CP_AIR

        gamma = 1./(1.-KAPPA)

        ptop_l = Field(data=dp[0])
        ptop_l = cal.unsqueeze(ptop_l, 0)
        ptop_l[:] = Field(data=ptop)

        pe = cal.cumsum(dp, dim=0) + ptop
        pe = cal.cat((ptop_l, pe))

        pln = cal.log(pe)
        dpln = cal.diff(pln)

        cs = cal.sqrt(-gamma*GRAV*dz/dpln)

        return cs
    
    @staticmethod
    @torch.jit.script
    def get_cs_sw(dp):

        cs = torch.sqrt(dp)

        return cs

    @staticmethod
    @torch.jit.script
    def get_pr_dyn(pt, dp, dz):
        #need to be accurate

        GRAV = 9.80665
        RDGAS = 287.05
        CP_AIR = 1004.6
        KAPPA = RDGAS/CP_AIR

        gamma = 1./(1.-KAPPA)

        rrg = -RDGAS/GRAV

        pr = torch.pow(dp*rrg*pt/dz, gamma)

        return pr
  
    @staticmethod
    @torch.jit.script
    def get_ph_dyn(dp, pe, ptop):

        RDGAS = 287.05
        CP_AIR = 1004.6
        KAPPA = RDGAS/CP_AIR
        
        pe[0,:,:] = ptop
        pe[1:,:,:] = dp
        pe[:] = torch.cumsum(pe, dim=0)
    
        pke = torch.pow(pe, KAPPA)
        dpk = torch.diff(pke, dim=0)

        gamma = 1./(1-KAPPA)

        return torch.pow( KAPPA*dp/dpk, gamma)

    @staticmethod
    @torch.jit.script
    def get_en_from_pt(en, rho, ua, va, wa, pt):

        RDGAS = 287.05
        CP_AIR = 1004.6
        KAPPA = RDGAS/CP_AIR
        
        Rd = RDGAS
        gamma = 1./(1.-KAPPA)

        en[:] = 0.5*(ua**2+va**2+wa**2) + torch.pow((Rd*pt*rho), gamma)/(gamma-1)/rho

        return
    
    @staticmethod
    @torch.jit.script
    def get_en_from_pr(en, rho, ua, va, wa, pr):

        RDGAS = 287.05
        CP_AIR = 1004.6
        KAPPA = RDGAS/CP_AIR

        gamma = 1./(1.-KAPPA)

        en[:] = 0.5*(ua**2+va**2+wa**2) + pr/(gamma-1)/rho

        return
    
    @staticmethod
    @torch.jit.script
    def get_pt_from_en(pt, rho, ua, va, wa, en):

        RDGAS = 287.05
        CP_AIR = 1004.6
        KAPPA = RDGAS/CP_AIR

        Rd = RDGAS
        gamma = 1./(1.-KAPPA)

        pt[:] = torch.pow(((en*rho-0.5*rho*(ua**2+va**2+wa**2))*(gamma-1)), (1.0/gamma)) / Rd / rho

        return
    
    @staticmethod
    @torch.jit.script
    def get_pr_from_en(pr, rho, ua, va, wa, en):

        RDGAS = 287.05
        CP_AIR = 1004.6
        KAPPA = RDGAS/CP_AIR

        gamma = 1./(1.-KAPPA)

        pr[:] = (en-0.5*(ua**2+va**2+wa**2)) * (gamma-1) * rho

        return
