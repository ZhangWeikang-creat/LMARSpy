
from lmarspy.backend.field import Field
from lmarspy.backend.cal import cal
from lmarspy.model.const import Const

import torch

class DynUtilVic:
    
    @staticmethod
    @torch.jit.script
    def calculate_res(res, rho, ua, va, wa, en,\
                      rho2, ua2, va2, wa2, en2, dt: float):
        
        res[:,:,:,0,0] = (rho2 - rho)/dt
        res[:,:,:,1,0] = (-rho2*ua2 + rho*ua)/dt
        res[:,:,:,2,0] = (rho2*va2 - rho*va)/dt
        res[:,:,:,3,0] = (rho2*wa2 - rho*wa)/dt
        res[:,:,:,4,0] = (rho2*en2 - rho*en)/dt

        return
    
    @staticmethod
    @torch.jit.script
    def get_var_tdex_reflect(rhoex, uex, vex, wex, pex,\
                             rho, u, v, w, p):

        uex[1:-1] = u
        uex[0] = u[0]
        uex[-1] = u[-1]

        vex[1:-1] = v
        vex[0] = v[0]
        vex[-1] = v[-1]

        wex[1:-1] = w
        wex[0] = -w[0]
        wex[-1] = -w[-1]
        
        pex[1:-1] = p
        pex[0] = p[0]
        pex[-1] = p[-1]

        rhoex[1:-1] = rho
        rhoex[0] = rho[0]
        rhoex[-1] = rho[-1]

        return
    
    
    @staticmethod
    @torch.jit.script
    def get_exact_jacobi(ja, rho, u, v, w, p):

        RDGAS = 287.05
        CP_AIR = 1004.6
        KAPPA = RDGAS/CP_AIR

        u = -u
        gamma = 1./(1.-KAPPA)
        gm1 = gamma - 1
        s2 = u**2+v**2+w**2
        c1 = u*((gamma-2)*s2/2.-gamma/gm1*p/rho)
        c2 = gamma/gm1*p/rho+s2/2-gm1*u**2

        ja[:,:,:,0,0] = 0
        ja[:,:,:,0,1] = 1
        ja[:,:,:,0,2] = 0
        ja[:,:,:,0,3] = 0
        ja[:,:,:,0,4] = 0

        ja[:,:,:,1,0] = gm1*s2/2-u**2
        ja[:,:,:,1,1] = (3-gamma)*u
        ja[:,:,:,1,2] = -gm1*v
        ja[:,:,:,1,3] = -gm1*w
        ja[:,:,:,1,4] =  gm1

        ja[:,:,:,2,0] = -u*v
        ja[:,:,:,2,1] = v
        ja[:,:,:,2,2] = u
        ja[:,:,:,2,3] = 0
        ja[:,:,:,2,4] = 0

        ja[:,:,:,3,0] = -u*w
        ja[:,:,:,3,1] =  w
        ja[:,:,:,3,2] = 0
        ja[:,:,:,3,3] = u
        ja[:,:,:,3,4] = 0

        ja[:,:,:,4,0] = c1
        ja[:,:,:,4,1] = c2
        ja[:,:,:,4,2] = -gm1*u*v
        ja[:,:,:,4,3] = -gm1*u*w
        ja[:,:,:,4,4] = gamma*u

        return 
    
    @staticmethod
    @torch.jit.script
    def get_roe_ave(rhor, ur, vr, wr, prr,\
                    rhoex, uex, vex, wex, prex): 

        rhor_tos = torch.sqrt(rhoex[1:])
        rhor_dns = torch.sqrt(rhoex[:-1])
        temp = 1. /(rhor_tos  + rhor_dns)
        rhor[1:] = rhor_tos * rhor_dns
        ur[1:] = (rhor_tos*uex[1:] + rhor_dns*uex[:-1]) * temp
        vr[1:] = (rhor_tos*vex[1:] + rhor_dns*vex[:-1]) * temp
        wr[1:] = (rhor_tos*wex[1:] + rhor_dns*wex[:-1]) * temp
        prr[1:] = (rhor_tos*prex[1:] + rhor_dns*prex[:-1]) * temp

        return
    
    @staticmethod
    @torch.jit.script
    def get_roe_jacobi(jr, pp, ee, ppinv, rhor0, ur0, vr0, wr0, prr0):

        RDGAS = 287.05
        CP_AIR = 1004.6
        KAPPA = RDGAS/CP_AIR

        rhor = rhor0[1:]
        ur = -ur0[1:]
        vr = vr0[1:]
        wr = wr0[1:]
        prr = prr0[1:]

        gamma = 1./(1.-KAPPA)
        gm1 = gamma - 1
        ker = (ur**2 + vr**2 + wr**2)*0.5
        hpr = gamma/gm1*prr/rhor
        hr = ker + hpr
        ar = cal.sqrt(gamma*prr/rhor)

        pp[:,:,:,0,0] = 1
        pp[:,:,:,0,1] = 1
        pp[:,:,:,0,2] = 1
        pp[:,:,:,0,3] = 0
        pp[:,:,:,0,4] = 0

        pp[:,:,:,1,0] = ur-ar
        pp[:,:,:,1,1] = ur
        pp[:,:,:,1,2] = ur+ar
        pp[:,:,:,1,3] = 0
        pp[:,:,:,1,4] = 0

        pp[:,:,:,2,0] = vr
        pp[:,:,:,2,1] = vr
        pp[:,:,:,2,2] = vr
        pp[:,:,:,2,3] = 1
        pp[:,:,:,2,4] = 0

        pp[:,:,:,3,0] = wr
        pp[:,:,:,3,1] = wr
        pp[:,:,:,3,2] = wr
        pp[:,:,:,3,3] = 0
        pp[:,:,:,3,4] = 1

        pp[:,:,:,4,0] = hr-ur*ar
        pp[:,:,:,4,1] = ker
        pp[:,:,:,4,2] = hr+ur*ar
        pp[:,:,:,4,3] =  vr
        pp[:,:,:,4,4] = wr

        ppinv[:,:,:,0,0] = (ar*ker+ur*hpr)/(2.*ar*hpr)
        ppinv[:,:,:,0,1] = (-hpr-ar*ur)/(2.*ar*hpr)
        ppinv[:,:,:,0,2] = -vr/(2.*hpr)
        ppinv[:,:,:,0,3] = -wr/(2.*hpr)
        ppinv[:,:,:,0,4] = 1./(2.*hpr)

        ppinv[:,:,:,1,0] = (hpr-ker)/hpr
        ppinv[:,:,:,1,1] = ur/hpr
        ppinv[:,:,:,1,2] = vr/hpr
        ppinv[:,:,:,1,3] =  wr/hpr
        ppinv[:,:,:,1,4] =  -1./hpr

        ppinv[:,:,:,2,0] = (ar*ker-ur*hpr)/(2.*ar*hpr)
        ppinv[:,:,:,2,1] = (hpr-ar*ur)/(2.*ar*hpr)
        ppinv[:,:,:,2,2] = -vr/(2.*hpr)
        ppinv[:,:,:,2,3] = -wr/(2.*hpr)
        ppinv[:,:,:,2,4] = 1./(2.*hpr)

        ppinv[:,:,:,3,0] = -vr
        ppinv[:,:,:,3,1] = 0
        ppinv[:,:,:,3,2] = 1
        ppinv[:,:,:,3,3] = 0
        ppinv[:,:,:,3,4] = 0

        ppinv[:,:,:,4,0] = -wr
        ppinv[:,:,:,4,1] = 0
        ppinv[:,:,:,4,2] = 0
        ppinv[:,:,:,4,3] = 1
        ppinv[:,:,:,4,4] = 0

        ee[:] = 0
        ee[:,:,:,0,0] = ur-ar
        ee[:,:,:,1,1] = ur
        ee[:,:,:,2,2] = ur+ar
        ee[:,:,:,3,3] = ur
        ee[:,:,:,4,4] = ur

        jr[:] = torch.matmul(torch.matmul(pp, torch.abs(ee)), ppinv)
        
        return

    @staticmethod
    def get_fz_vic(delta_R, ja, jr, res, dt: float, dz, mp):

        npz = mp.npz
        
        expanded_dz = cal.unsqueeze(cal.unsqueeze(-dz, dim=-1), dim=-1)

        grav = 9.80665
        DtX = [[1./dt,  0.,    0.,    0.,     0.],
               [-grav,  1./dt,    0.,    0.,     0.],
               [0.,     0., 1./dt,    0.,     0.],
               [0.,  0.,    0., 1./dt,     0.],
               [0.,     -grav,    0.,    0.,  1./dt]]
        DtX = Field(data=DtX, dty="float64")

        M = [[1., 0., 0.,  0.,  0.],
             [0.,-1., 0.,  0.,  0.],
             [0., 0., 1.,  0.,  0.],
             [0., 0., 0.,  1.,  0.],
             [0., 0., 0.,  0.,  1.]]
        M = Field(data=M, dty="float64")

        bi, ai, ci = DynUtilVic.get_ai_bi_ci(ja, jr, expanded_dz, DtX, M)

        DynUtilVic.solve_diagonal_equations(delta_R, bi, ai, ci, res, npz)

        return 
    
    @staticmethod
    @torch.jit.script
    def get_ai_bi_ci(ja, jr, expanded_dz, DtX, M):

        
        ai = 0.5*(jr[:-1] + jr[1:])/expanded_dz + DtX
        bi = -0.5*(jr[:-1] + ja[:-2])/expanded_dz
        ci = -0.5*(jr[1:] - ja[2:])/expanded_dz

        ai[0] += torch.matmul(bi[0] , M)
        ai[-1] += torch.matmul(ci[-1] , M)

        return bi, ai, ci
    
    @staticmethod
    @torch.jit.script
    def solve_diagonal_equations(delta_R, bi, ai, ci, res, npz:int):

        k = 0
        ai[k] = torch.linalg.inv(ai[k])
        delta_R[k] = torch.matmul(ai[k],res[k])
        ai[k] = torch.matmul(ai[k], ci[k])
        for k in range(1, npz):
            ai[k] = torch.linalg.inv(ai[k] - torch.matmul(bi[k], ai[k-1]))
            delta_R[k] = torch.matmul(ai[k],res[k]-torch.matmul(bi[k],delta_R[k-1]))
            ai[k] = torch.matmul(ai[k], ci[k])
        for k in range(npz-2, -1, -1):
            delta_R[k] -= torch.matmul(ai[k],delta_R[k+1])

        return
    
    @staticmethod
    @torch.jit.script
    def update_vic_var(delta_R, rho0, ua0, va0, wa0, en0,\
                         rho, ua, va, wa, en):

        rho[:] = rho0 + delta_R[:,:,:,0,0]
        ua[:] = (ua0*rho0 - delta_R[:,:,:,1,0])/rho
        va[:] = (va0*rho0 + delta_R[:,:,:,2,0])/rho
        wa[:] = (wa0*rho0 + delta_R[:,:,:,3,0])/rho
        en[:] = (en0*rho0 + delta_R[:,:,:,4,0])/rho

        return
    
