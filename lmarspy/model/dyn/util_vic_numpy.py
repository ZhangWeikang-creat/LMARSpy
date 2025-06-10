
from lmarspy.backend.field import Field
from lmarspy.backend.cal import cal
from lmarspy.model.const import Const
import numpy as np
from numba import jit

class DynUtilVic:
    
    @staticmethod
    @jit(nopython=True)
    def calculate_res(res, rho, ua, va, wa, pt,\
                      rho2, ua2, va2, wa2, pt2, dt: float):
        
        res[:,:,:,0,0] = (rho2 - rho)/dt
        res[:,:,:,1,0] = (rho2*ua2 - rho*ua)/dt
        res[:,:,:,2,0] = (rho2*va2 - rho*va)/dt
        res[:,:,:,3,0] = (-rho2*wa2 + rho*wa)/dt
        res[:,:,:,4,0] = (rho2*pt2 - rho*pt)/dt

        return
    
    @staticmethod
    @jit(nopython=True)
    def get_var_tdex_reflect(rhoex, uex, vex, wex, ptex,\
                             rho, u, v, w, pt):

        uex[1:-1] = u
        uex[0] = u[0]
        uex[-1] = u[-1]

        vex[1:-1] = v
        vex[0] = v[0]
        vex[-1] = v[-1]

        wex[1:-1] = w
        wex[0] = -w[0]
        wex[-1] = -w[-1]
        
        ptex[1:-1] = pt
        ptex[0] = pt[0]
        ptex[-1] = pt[-1]

        rhoex[1:-1] = rho
        rhoex[0] = rho[0]
        rhoex[-1] = rho[-1]

        return
    
    
    @staticmethod
    @jit(nopython=True)
    def get_exact_jacobi(ja, rho, u, v, w0, pt):
        w = -w0

        RDGAS = 287.05
        CP_AIR = 1004.6
        KAPPA = RDGAS/CP_AIR
        gm = 1./(1.-KAPPA)
        R = RDGAS

        ja[:,:,:,0,0] = 0
        ja[:,:,:,0,1] = 0
        ja[:,:,:,0,2] = 0
        ja[:,:,:,0,3] = 1
        ja[:,:,:,0,4] = 0

        ja[:,:,:,1,0] = -u*w
        ja[:,:,:,1,1] = w
        ja[:,:,:,1,2] = 0
        ja[:,:,:,1,3] = u
        ja[:,:,:,1,4] = 0

        ja[:,:,:,2,0] = -v*w
        ja[:,:,:,2,1] = 0
        ja[:,:,:,2,2] = w
        ja[:,:,:,2,3] = v
        ja[:,:,:,2,4] = 0

        ja[:,:,:,3,0] = -w**2
        ja[:,:,:,3,1] = 0
        ja[:,:,:,3,2] = 0
        ja[:,:,:,3,3] = 2*w
        ja[:,:,:,3,4] = R*gm*(R*pt*rho)**(gm - 1)

        ja[:,:,:,4,0] = -pt*w
        ja[:,:,:,4,1] = 0
        ja[:,:,:,4,2] = 0
        ja[:,:,:,4,3] = pt
        ja[:,:,:,4,4] = w

        return 
    
    @staticmethod
    @jit(nopython=True)
    def get_roe_ave(rhor, ur, vr, wr, ptr,\
                    rhoex, uex, vex, wex, ptex): 

        rhor_tos = np.sqrt(rhoex[1:])
        rhor_dns = np.sqrt(rhoex[:-1])
        temp = 1. /(rhor_tos  + rhor_dns)
        rhor[1:] = rhor_tos * rhor_dns
        ur[1:] = (rhor_tos*uex[1:] + rhor_dns*uex[:-1]) * temp
        vr[1:] = (rhor_tos*vex[1:] + rhor_dns*vex[:-1]) * temp
        wr[1:] = (rhor_tos*wex[1:] + rhor_dns*wex[:-1]) * temp
        ptr[1:] = (rhor_tos*ptex[1:] + rhor_dns*ptex[:-1]) * temp

        return
    
    @staticmethod
    @jit(nopython=True)
    def get_roe_jacobi(jr, rhor0, ur0, vr0, wr0, ptr0):

        RDGAS = 287.05
        CP_AIR = 1004.6
        KAPPA = RDGAS/CP_AIR

        rho = rhor0[1:]
        u = ur0[1:]
        v = vr0[1:]
        w = -wr0[1:]
        pt = ptr0[1:]

        R = RDGAS
        gm = 1./(1.-KAPPA)
        a = (gm*(R*pt*rho)**gm/rho)**(1/2)
        wa = np.abs(w)
        wma = a-w
        wpa = w+a

        jr[:,:,:,0,0] = wa + (w*wma)/(2*a) - (w*wpa)/(2*a)
        jr[:,:,:,0,1] = 0
        jr[:,:,:,0,2] = 0
        jr[:,:,:,0,3] = wpa/(2*a) - wma/(2*a)
        jr[:,:,:,0,4] = wma/(2*pt) - wa/pt + wpa/(2*pt)

        jr[:,:,:,1,0] = (u*w*wma)/(2*a) - (u*w*wpa)/(2*a)
        jr[:,:,:,1,1] = wa
        jr[:,:,:,1,2] = 0
        jr[:,:,:,1,3] = (u*wpa)/(2*a) - (u*wma)/(2*a)
        jr[:,:,:,1,4] = (u*wma)/(2*pt) - (u*wa)/pt + (u*wpa)/(2*pt)

        jr[:,:,:,2,0] = (v*w*wma)/(2*a) - (v*w*wpa)/(2*a)
        jr[:,:,:,2,1] = 0
        jr[:,:,:,2,2] = wa
        jr[:,:,:,2,3] = (v*wpa)/(2*a) - (v*wma)/(2*a)
        jr[:,:,:,2,4] = (v*wma)/(2*pt) - (v*wa)/pt + (v*wpa)/(2*pt)

        jr[:,:,:,3,0] = w*wa - (w*wpa*(a + w))/(2*a) - (w*wma*(a - w))/(2*a)
        jr[:,:,:,3,1] = 0
        jr[:,:,:,3,2] = 0
        jr[:,:,:,3,3] = (wma*(a - w))/(2*a) + (wpa*(a + w))/(2*a)
        jr[:,:,:,3,4] = (wpa*(a + w))/(2*pt) - (wma*(a - w))/(2*pt) - (w*wa)/pt

        jr[:,:,:,4,0] = (pt*w*wma)/(2*a) - (pt*w*wpa)/(2*a)
        jr[:,:,:,4,1] = 0
        jr[:,:,:,4,2] = 0
        jr[:,:,:,4,3] = (pt*wpa)/(2*a) - (pt*wma)/(2*a)
        jr[:,:,:,4,4] = wma/2 + wpa/2
        
        
        return

    @staticmethod
    def get_fz_vic(delta_R, ja, jr, res, dt: float, dz, mp):

        npx = mp.npx
        npy = mp.npy
        npz = mp.npz
        expanded_dz = cal.unsqueeze(cal.unsqueeze(-dz, dim=-1), dim=-1)
        
        bi, ai, ci = DynUtilVic.get_ai_bi_ci(ja, jr, expanded_dz, dt, npx, npy)

        DynUtilVic.solve_diagonal_equations(delta_R, bi, ai, ci, res, npz)

        return 
    
    @staticmethod
    @jit(nopython=True)
    def get_ai_bi_ci(ja, jr, expanded_dz, dt: float, npx: int, npy: int):

        grav = 9.80665
        DtX = [[1./dt,  0.,    0.,    0.,     0.],
               [0.,  1./dt,    0.,    0.,     0.],
               [0.,     0., 1./dt,    0.,     0.],
               [-grav,  0.,    0., 1./dt,     0.],
               [0.,     0.,    0.,    0.,  1./dt]]
        DtX = np.array(DtX, dtype=np.float64)
        M = [[1., 0., 0.,  0.,  0.],
             [0., 1., 0.,  0.,  0.],
             [0., 0., 1.,  0.,  0.],
             [0., 0., 0., -1.,  0.],
             [0., 0., 0.,  0.,  1.]]
        M = np.array(M, dtype=np.float64)
        ai = 0.5*(jr[:-1] + jr[1:])/expanded_dz + DtX
        bi = -0.5*(jr[:-1] + ja[:-2])/expanded_dz
        ci = -0.5*(jr[1:] - ja[2:])/expanded_dz

        def matmul_batch(A, B):
            rlt = np.zeros_like(A, dtype=np.float64)
            n = 5
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        rlt[:,:,i,j] += A[:,:,i,k]*B[k,j]
            return rlt

        ai[0] += matmul_batch(bi[0], M)
        ai[-1] += matmul_batch(ci[-1], M)

        return bi, ai, ci
    
    '''
    @staticmethod
    @jit(nopython=True)
    def solve_diagonal_equations(delta_R, bi, ai, ci, res, npz: int):
        
        def matrix_inverse(A):
            npy, npx, m, n = A.shape
            augmented_matrix = np.zeros((npy, npx, n, n*2), dtype=np.float64)
            augmented_matrix[:,:,:,:n] = A
            augmented_matrix[:,:,:,n:] = np.eye(n, dtype=np.float64)
            for i in range(n):
                
                # 没有完全并行化（但cpu可能不需要优化）
                max_row = np.argmax(np.abs(augmented_matrix[:,:, i:n, i]), axis=-1) + i
                for nx in range(npx):
                    for ny in range(npy):
                        temp = augmented_matrix[ny,nx, i,:].copy()
                        augmented_matrix[ny,nx,i,:]= augmented_matrix[ny,nx,max_row[ny,nx],:]
                        augmented_matrix[ny,nx,max_row[ny,nx],:]= temp

                augmented_matrix[:,:, i,:] = augmented_matrix[:,:, i,:] / augmented_matrix[:,:, i, i][:, :,None]

                for j in range(n):
                    if j != i:
                        augmented_matrix[:,:, j,:] -=  augmented_matrix[:,:,  j, i][:, :,None] * augmented_matrix[:,:, i,:]
            
            return augmented_matrix[:,:, :, n:]

        def matmul_batch1(A, B):
            rlt = np.zeros_like(A, dtype=np.float64)
            n = 5
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        rlt[:,:,i,j] += A[:,:,i,k]*B[:,:,k,j]

            return rlt
        
        def matmul_batch2(A, b):
            rlt = np.zeros_like(b, dtype=np.float64)
            n = 5
            for i in range(n):
                for k in range(n):
                    rlt[:,:,i,0] += A[:,:,i,k]*b[:,:,k,0]

            return rlt
        
        k = 0
        ai[k] = matrix_inverse(ai[k])
        delta_R[k] = matmul_batch2(ai[k],res[k])
        ai[k] = matmul_batch1(ai[k], ci[k])
        for k in range(1, npz):
            ai[k] = matrix_inverse(ai[k] - matmul_batch1(bi[k], ai[k-1]))
            delta_R[k] = matmul_batch2(ai[k],res[k]-matmul_batch2(bi[k],delta_R[k-1]))
            ai[k] = matmul_batch1(ai[k], ci[k])
        for k in range(npz-2, -1, -1):
            delta_R[k] = delta_R[k] - matmul_batch2(ai[k],delta_R[k+1])

        return
    '''

    @staticmethod
    def solve_diagonal_equations(delta_R, bi, ai, ci, res, npz:int):

        k = 0
        ai[k] = np.linalg.inv(ai[k])
        delta_R[k] = np.matmul(ai[k],res[k])
        ai[k] = np.matmul(ai[k], ci[k])
        for k in range(1, npz):
            ai[k] = np.linalg.inv(ai[k] - np.matmul(bi[k], ai[k-1]))
            delta_R[k] = np.matmul(ai[k],res[k]-np.matmul(bi[k],delta_R[k-1]))
            ai[k] = np.matmul(ai[k], ci[k])
        for k in range(npz-2, -1, -1):
            delta_R[k] = delta_R[k] - np.matmul(ai[k],delta_R[k+1])

        return

    
    @staticmethod
    @jit(nopython=True)
    def update_vic_var(delta_R, rho0, ua0, va0, wa0, pt0,\
                         rho, ua, va, wa, pt):

        rho[:] = rho0 + delta_R[:,:,:,0,0]
        ua[:] = (ua0*rho0 + delta_R[:,:,:,1,0])/rho
        va[:] = (va0*rho0 + delta_R[:,:,:,2,0])/rho
        wa[:] = -(-wa0*rho0 + delta_R[:,:,:,3,0])/rho
        pt[:] = (pt0*rho0 + delta_R[:,:,:,4,0])/rho

        return
    
