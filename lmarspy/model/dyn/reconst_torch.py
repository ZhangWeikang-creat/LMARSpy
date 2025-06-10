#!/usr/bin/env python

import torch

#--- reconst lib

class DynRc:

    @staticmethod
    @torch.jit.script
    def get_var_we_ng1(pa, pw, pe) -> None:

        pw[:] = pa[:,:,1:]
        pe[:] = pa[:,:,:-1]

        return

    @staticmethod
    @torch.jit.script
    def get_var_we_ng2(pa, pw, pe) -> None:

        E2m1, E2c0, E2p1 = -1. / 6., 5. / 6., 1. / 3.
        W2m1, W2c0, W2p1 = E2p1, E2c0, E2m1

        y1, y2, y3 = pa[:,:,:-2], pa[:,:,1:-1], pa[:,:,2:]

        pwt = W2m1*y1 + W2c0*y2 + W2p1*y3
        pet = E2m1*y1 + E2c0*y2 + E2p1*y3

        pw[:] = pwt[:,:,1:]
        pe[:] = pet[:,:,:-1]

        return
    
    @staticmethod
    @torch.jit.script
    def get_var_we_lim_ng2(pa, pw, pe, lim_deg: float) -> None:

        E2m1, E2c0, E2p1 = -1. / 6., 5. / 6., 1. / 3.
        W2m1, W2c0, W2p1 = E2p1, E2c0, E2m1

        vare = 1E-10

        y1, y2, y3 = pa[:,:,:-2], pa[:,:,1:-1], pa[:,:,2:]

        left_low = W2m1*y1 + W2c0*y2 + W2p1*y3
        right_low = E2m1*y1 + E2c0*y2 + E2p1*y3

        r = (y2-y1)/(y3-y2+vare)
        sigma = (r*2)/(1+r**2)
        sigma = torch.where(sigma > (1-lim_deg), sigma, 1-lim_deg)

        pwt = y2 - sigma*(y2-left_low)
        pet = y2 - sigma*(y2-right_low)

        pw[:] = pwt[:,:,1:]
        pe[:] = pet[:,:,:-1]

        return

    @staticmethod
    @torch.jit.script
    def get_var_we_ng3(pa, pw, pe) -> None:

        E3m2, E3m1, E3c0, E3p1, E3p2 = 1. / 30., -13. / 60., 47. / 60., 9. / 20., -1. / 20.
        W3m2, W3m1, W3c0, W3p1, W3p2 = E3p2, E3p1, E3c0, E3m1, E3m2

        y1, y2, y3, y4, y5 = pa[:, :, :-4], pa[:, :, 1:-3], pa[:, :, 2:-2], pa[:, :, 3:-1], pa[:, :, 4:]

        pwt = W3m2*y1 + W3m1*y2 + W3c0*y3 + W3p1*y4 + W3p2*y5
        pet = E3m2*y1 + E3m1*y2 + E3c0*y3 + E3p1*y4 + E3p2*y5

        pw[:] = pwt[:,:,1:]
        pe[:] = pet[:,:,:-1]
        
        return
    
    @staticmethod
    @torch.jit.script
    def get_var_we_lim_ng3(pa, pw, pe, lim_deg: float) -> None:

        E2m1, E2c0, E2p1 = -1. / 6., 5. / 6., 1. / 3.
        W2m1, W2c0, W2p1 = E2p1, E2c0, E2m1
        
        E3m2, E3m1, E3c0, E3p1, E3p2 = 1. / 30., -13. / 60., 47. / 60., 9. / 20., -1. / 20.
        W3m2, W3m1, W3c0, W3p1, W3p2 = E3p2, E3p1, E3c0, E3m1, E3m2

        vare = 1E-10

        y1, y2, y3, y4, y5 = pa[:, :, :-4], pa[:, :, 1:-3], pa[:, :, 2:-2], pa[:, :, 3:-1], pa[:, :, 4:]

        left_low = W2m1*y2 + W2c0*y3 + W2p1*y4
        right_low = E2m1*y2 + E2c0*y3 + E2p1*y4

        left_high = W3m2*y1 + W3m1*y2 + W3c0*y3 + W3p1*y4 + W3p2*y5
        right_high = E3m2*y1 + E3m1*y2 + E3c0*y3 + E3p1*y4 + E3p2*y5
        
        r = (y3-y2)/(y4-y3+vare)
        sigma = (r*2)/(1+r**2)

        sigma = torch.where(sigma > (1-lim_deg), sigma, 1-lim_deg)

        temp1 = y3 - sigma*(y3-left_low)
        temp2 = y3 - sigma*(y3-right_low)

        pwt = temp1 - sigma*(temp1 - left_high)
        pet = temp2 - sigma*(temp2 - right_high)

        pw[:] = pwt[:,:,1:]
        pe[:] = pet[:,:,:-1]
        
        return
    
    @staticmethod
    @torch.jit.script
    def get_var_sn_ng1(pa, ps, pn) -> None:

        ps[:] = pa[:,1:,:]
        pn[:] = pa[:,:-1,:]

        return

    @staticmethod
    @torch.jit.script
    def get_var_sn_ng2(pa, ps, pn) -> None:

        E2m1, E2c0, E2p1 = -1. / 6., 5. / 6., 1. / 3.
        W2m1, W2c0, W2p1 = E2p1, E2c0, E2m1

        y1, y2, y3 = pa[:,:-2,:], pa[:,1:-1,:], pa[:,2:,:]

        pst = W2m1*y1 + W2c0*y2 + W2p1*y3
        pnt = E2m1*y1 + E2c0*y2 + E2p1*y3

        ps[:] = pst[:,1:,:]
        pn[:] = pnt[:,:-1,:]
        
        return
    
    @staticmethod
    @torch.jit.script
    def get_var_sn_lim_ng2(pa, ps, pn, lim_deg: float) -> None:

        E2m1, E2c0, E2p1 = -1. / 6., 5. / 6., 1. / 3.
        W2m1, W2c0, W2p1 = E2p1, E2c0, E2m1

        y1, y2, y3 = pa[:,:-2,:], pa[:,1:-1,:], pa[:,2:,:]

        vare = 1E-10

        left_low = W2m1*y1 + W2c0*y2 + W2p1*y3
        right_low = E2m1*y1 + E2c0*y2 + E2p1*y3

        r = (y2-y1)/(y3-y2+vare)
        sigma = (r*2)/(1+r**2)
        sigma = torch.where(sigma > (1-lim_deg), sigma, 1-lim_deg)

        pst = y2 - sigma*(y2-left_low)
        pnt = y2 - sigma*(y2-right_low)

        ps[:] = pst[:,1:,:]
        pn[:] = pnt[:,:-1,:]
        
        return
    

    @staticmethod
    @torch.jit.script
    def get_var_sn_ng3(pa, ps, pn) -> None:

        E3m2, E3m1, E3c0, E3p1, E3p2 = 1. / 30., -13. / 60., 47. / 60., 9. / 20., -1. / 20.
        W3m2, W3m1, W3c0, W3p1, W3p2 = E3p2, E3p1, E3c0, E3m1, E3m2

        y1, y2, y3, y4, y5= pa[:,:-4,:], pa[:,1:-3,:], pa[:,2:-2,:], pa[:,3:-1,:], pa[:,4:,:]

        pst = W3m2*y1 + W3m1*y2 + W3c0*y3 + W3p1*y4 + W3p2*y5
        pnt = E3m2*y1 + E3m1*y2 + E3c0*y3 + E3p1*y4 + E3p2*y5

        ps[:] = pst[:,1:,:]
        pn[:] = pnt[:,:-1,:]
        
        return
    
    @staticmethod
    @torch.jit.script
    def get_var_sn_lim_ng3(pa, ps, pn, lim_deg: float) -> None:

        E2m1, E2c0, E2p1 = -1. / 6., 5. / 6., 1. / 3.
        W2m1, W2c0, W2p1 = E2p1, E2c0, E2m1

        E3m2, E3m1, E3c0, E3p1, E3p2 = 1. / 30., -13. / 60., 47. / 60., 9. / 20., -1. / 20.
        W3m2, W3m1, W3c0, W3p1, W3p2 = E3p2, E3p1, E3c0, E3m1, E3m2

        vare = 1E-10

        y1, y2, y3, y4, y5= pa[:,:-4,:], pa[:,1:-3,:], pa[:,2:-2,:], pa[:,3:-1,:], pa[:,4:,:]

        left_low = W2m1*y2 + W2c0*y3 + W2p1*y4
        right_low = E2m1*y2 + E2c0*y3 + E2p1*y4

        left_high = W3m2*y1 + W3m1*y2 + W3c0*y3 + W3p1*y4 + W3p2*y5
        right_high = E3m2*y1 + E3m1*y2 + E3c0*y3 + E3p1*y4 + E3p2*y5
        
        r = (y3-y2)/(y4-y3+vare)
        sigma = (r*2)/(1+r**2)
        sigma = torch.where(sigma > (1-lim_deg), sigma, 1-lim_deg)

        temp1 = y3 - sigma*(y3-left_low)
        temp2 = y3 - sigma*(y3-right_low)

        pst = temp1 - sigma*(temp1 - left_high)
        pnt = temp2 - sigma*(temp2 - right_high)

        ps[:] = pst[:,1:,:]
        pn[:] = pnt[:,:-1,:]
        
        return

    @staticmethod
    @torch.jit.script
    def get_var_td_ng1(pa, pto, pdn) -> None:

        pto[:,:,:] = pa[:,:,:]
        pdn[:,:,:] = pa[:,:,:]

        return

    @staticmethod
    @torch.jit.script
    def get_var_td_ng2(pa, pto, pdn) -> None:

        E2m1, E2c0, E2p1 = -1. / 6., 5. / 6., 1. / 3.
        W2m1, W2c0, W2p1 = E2p1, E2c0, E2m1

        y1, y2, y3 = pa[:-2,:,:], pa[1:-1,:,:], pa[2:,:,:]

        pto[1:-1,:,:] = W2m1*y1 + W2c0*y2 + W2p1*y3
        pdn[1:-1,:,:] = E2m1*y1 + E2c0*y2 + E2p1*y3

        #cal.get_var_td_edge0_o2(pa, pto, pdn)

        W0, Wp1, E0, Ep1 = 1.5, -0.5, 0.5, 0.5

        pto[0,:,:] = W0*pa[0,:,:] + Wp1*pa[1,:,:]
        pdn[0,:,:] = E0*pa[0,:,:] + Ep1*pa[1,:,:]

        pto[-1,:,:] = E0*pa[-1,:,:] + Ep1*pa[-2,:,:]
        pdn[-1,:,:] = W0*pa[-1,:,:] + Wp1*pa[-2,:,:]

        return
    
    @staticmethod
    @torch.jit.script
    def get_var_td_lim_ng2(pa, pto, pdn, lim_deg: float) -> None:

        E2m1, E2c0, E2p1 = -1. / 6., 5. / 6., 1. / 3.
        W2m1, W2c0, W2p1 = E2p1, E2c0, E2m1

        vare = 1E-10

        y1, y2, y3 = pa[:-2,:,:], pa[1:-1,:,:], pa[2:,:,:]

        left_low = W2m1*y1 + W2c0*y2 + W2p1*y3
        right_low = E2m1*y1 + E2c0*y2 + E2p1*y3

        r = (y2-y1)/(y3-y2+vare)
        sigma = (r*2)/(1+r**2)
        sigma = torch.where(sigma > (1-lim_deg), sigma, 1-lim_deg)

        pto[1:-1,:,:] = y2 - sigma*(y2-left_low)
        pdn[1:-1,:,:] = y2 - sigma*(y2-right_low)

        #cal.get_var_td_edge0_o2(pa, pto, pdn)

        W0, Wp1, E0, Ep1 = 1.5, -0.5, 0.5, 0.5

        pto[0,:,:] = W0*pa[0,:,:] + Wp1*pa[1,:,:]
        pdn[0,:,:] = E0*pa[0,:,:] + Ep1*pa[1,:,:]

        pto[-1,:,:] = E0*pa[-1,:,:] + Ep1*pa[-2,:,:]
        pdn[-1,:,:] = W0*pa[-1,:,:] + Wp1*pa[-2,:,:]

        return


    @staticmethod
    @torch.jit.script
    def get_var_td_ng3(pa, pto, pdn) -> None:

        E3m2, E3m1, E3c0, E3p1, E3p2 = 1. / 30., -13. / 60., 47. / 60., 9. / 20., -1. / 20.
        W3m2, W3m1, W3c0, W3p1, W3p2 = E3p2, E3p1, E3c0, E3m1, E3m2

        y1, y2, y3, y4, y5= pa[:-4,:,:], pa[1:-3,:,:], pa[2:-2,:,:], pa[3:-1,:,:], pa[4:,:,:]

        pto[2:-2,:,:] = W3m2*y1 + W3m1*y2 + W3c0*y3 + W3p1*y4 + W3p2*y5
        pdn[2:-2,:,:] = E3m2*y1 + E3m1*y2 + E3c0*y3 + E3p1*y4 + E3p2*y5

        #cal.get_var_td_edge1_o3(pa, pto, pdn)

        E2m1, E2c0, E2p1 = -1. / 6., 5. / 6., 1. / 3.
        W2m1, W2c0, W2p1 = E2p1, E2c0, E2m1

        pto[1,:,:] = W2m1*pa[0,:,:] + W2c0*pa[1,:,:] + W2p1*pa[2,:,:]
        pdn[1,:,:] = E2m1*pa[0,:,:] + E2c0*pa[1,:,:] + E2p1*pa[2,:,:]

        pto[-2,:,:] = W2m1*pa[-3,:,:] + W2c0*pa[-2,:,:] + W2p1*pa[-1,:,:]
        pdn[-2,:,:] = E2m1*pa[-3,:,:] + E2c0*pa[-2,:,:] + E2p1*pa[-1,:,:]

        #cal.get_var_td_edge0_o2(pa, pto, pdn)

        W0, Wp1, E0, Ep1 = 1.5, -0.5, 0.5, 0.5

        pto[0,:,:] = W0*pa[0,:,:] + Wp1*pa[1,:,:]
        pdn[0,:,:] = E0*pa[0,:,:] + Ep1*pa[1,:,:]

        pto[-1,:,:] = E0*pa[-1,:,:] + Ep1*pa[-2,:,:]
        pdn[-1,:,:] = W0*pa[-1,:,:] + Wp1*pa[-2,:,:]

        return
    
    @staticmethod
    @torch.jit.script
    def get_var_td_lim_ng3(pa, pto, pdn, lim_deg: float) -> None:

        E2m1, E2c0, E2p1 = -1. / 6., 5. / 6., 1. / 3.
        W2m1, W2c0, W2p1 = E2p1, E2c0, E2m1

        E3m2, E3m1, E3c0, E3p1, E3p2 = 1. / 30., -13. / 60., 47. / 60., 9. / 20., -1. / 20.
        W3m2, W3m1, W3c0, W3p1, W3p2 = E3p2, E3p1, E3c0, E3m1, E3m2

        vare = 1E-10

        y1, y2, y3, y4, y5= pa[:-4,:,:], pa[1:-3,:,:], pa[2:-2,:,:], pa[3:-1,:,:], pa[4:,:,:]

        left_low = W2m1*y2 + W2c0*y3 + W2p1*y4
        right_low = E2m1*y2 + E2c0*y3 + E2p1*y4

        left_high = W3m2*y1 + W3m1*y2 + W3c0*y3 + W3p1*y4 + W3p2*y5
        right_high = E3m2*y1 + E3m1*y2 + E3c0*y3 + E3p1*y4 + E3p2*y5
        
        r = (y3-y2)/(y4-y3+vare)
        sigma = (r*2)/(1+r**2)
        sigma = torch.where(sigma > (1-lim_deg), sigma, 1-lim_deg)

        temp1 = y3 - sigma*(y3-left_low)
        temp2 = y3 - sigma*(y3-right_low)

        pto[2:-2,:,:] = temp1 - sigma*(temp1 - left_high)
        pdn[2:-2,:,:] = temp2 - sigma*(temp2 - right_high)

        #cal.get_var_td_edge0_o2(pa, pto, pdn)

        W0, Wp1, E0, Ep1 = 1.5, -0.5, 0.5, 0.5

        pto[0,:,:] = W0*pa[0,:,:] + Wp1*pa[1,:,:]
        pdn[0,:,:] = E0*pa[0,:,:] + Ep1*pa[1,:,:]

        pto[-1,:,:] = E0*pa[-1,:,:] + Ep1*pa[-2,:,:]
        pdn[-1,:,:] = W0*pa[-1,:,:] + Wp1*pa[-2,:,:]

        #cal.get_var_td_lim_edge1_o3(pa, pto, pdn)

        y1, y2, y3 = pa[0,:,:], pa[1,:,:], pa[2,:,:]

        left_low = W2m1*y1 + W2c0*y2 + W2p1*y3
        right_low = E2m1*y1 + E2c0*y2 + E2p1*y3

        r = (y2-y1)/(y3-y2+vare)
        sigma = (r*2)/(1+r**2)
        sigma = torch.where(sigma > (1-lim_deg), sigma, 1-lim_deg)

        pto[1,:,:] = y2 - sigma*(y2-left_low)
        pdn[1,:,:] = y2 - sigma*(y2-right_low)

        y1, y2, y3 = pa[-3,:,:], pa[-2,:,:], pa[-1,:,:]
        
        left_low = W2m1*y1 + W2c0*y2 + W2p1*y3
        right_low = E2m1*y1 + E2c0*y2 + E2p1*y3

        r = (y2-y1)/(y3-y2+vare)
        sigma = (r*2)/(1+r**2)
        sigma = torch.where(sigma > (1-lim_deg), sigma, 1-lim_deg)

        pto[-2,:,:] = y2 - sigma*(y2-left_low)
        pdn[-2,:,:] = y2 - sigma*(y2-right_low)

        return
    

