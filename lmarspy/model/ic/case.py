import numpy as np
from .data import ICdata

RADIUS = 6.3712e6
PI = 3.1415926535897931
OMEGA = 7.2921e-5
GRAV = 9.80665
RDGAS = 287.05
RVGAS = 461.50
CP_AIR = 1004.6
KAPPA = RDGAS/CP_AIR

class ICcase:
    
    @staticmethod
    def sw_gauss_wave(IC: ICdata):
        #--- dims
        npx = IC.npx
        npy = IC.npy
        npz = IC.npz
        ng  = IC.ng

        ibd = IC.ibd
        ib  = IC.ib
        ie  = IC.ie
        ied = IC.ied

        jbd = IC.jbd
        jb  = IC.jb
        je  = IC.je
        jed = IC.jed

        # get dx, dy, dz
        nx = IC.global_npx - 1
        ny = IC.global_npy - 1
        nz = npz

        domain_size = 20000
        dx = domain_size/nx
        dy = domain_size/ny
        dz = domain_size/nz

        IC.dx = dx
        IC.dy = dy
        IC.dz = dz

        if IC.global_npx == 2:
            global_x = (np.arange(1)+0.5) * dx
            IC.global_x = global_x
            x = global_x
            IC.x = x

        else:
            global_x = (np.arange(IC.global_npx+ng*2-1) + 0.5-IC.ib) * dx
            IC.global_x = global_x[ng:-ng]
            x = global_x[IC.global_ibd:IC.global_ied]
            IC.x = x[ng:-ng]

        if IC.global_npy == 2:
            global_y = (np.arange(1)+0.5) * dy
            IC.global_y = global_y
            y = global_y
            IC.y = y

        else:
            global_y = (np.arange(IC.global_npy+ng*2-1) + 0.5-IC.jb) * dy
            IC.global_y = global_y[ng:-ng]
            y = global_y[IC.global_jbd:IC.global_jed]
            IC.y = y[ng:-ng]

        z = 20000 - (np.arange(nz) + 0.5) * dz
        IC.z = z

        a_x = np.expand_dims(x[:], axis=0)
        a_dx = dx
        a_rdx = 1./a_dx

        a_y = np.expand_dims(y[:], axis=1)
        a_dy = dy
        a_rdy = 1./a_dy

        a_da = a_dx*a_dy
        a_rda = 1./a_da

        d_dx = dx
        c_dy = dy

        IC.a_x = a_x
        IC.a_dx = a_dx
        IC.a_rdx = a_rdx

        IC.a_y = a_y
        IC.a_dy = a_dy
        IC.a_rdy = a_rdy

        IC.a_da = a_da
        IC.a_rda = a_rda

        IC.d_dx = d_dx
        IC.c_dy = c_dy

        #--- progs
        delp = IC.delp
        delz = IC.delz
        ua = IC.ua
        va = IC.va
        wa = IC.wa
        pt = IC.pt

        #--- aux
        ps = IC.ps
        phis = IC.phis
        ptop = 0

        delz[:] = -dz

        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        IC.xx = xx
        IC.yy = yy
        IC.zz = zz

        IC.ptop = 0

        #--- delp, ua, pt
        pt[:] = 0
        
        d = 10
        wave_center = 10000
        delp[0,:,:] = d*GRAV
        temp = np.sqrt((xx-wave_center)**2 + (yy-wave_center)**2)/domain_size*24

        delp[:,:,:] += np.exp(-temp**2/2)*d*GRAV*0.1

        delz[:] = -delp/GRAV # assume rho=1
        ua[:] = 0
        va[:] = 0

        return
    
    @staticmethod
    def sw_square_wave(IC: ICdata):
        #--- dims
        npx = IC.npx
        npy = IC.npy
        npz = IC.npz
        ng  = IC.ng

        ibd = IC.ibd
        ib  = IC.ib
        ie  = IC.ie
        ied = IC.ied

        jbd = IC.jbd
        jb  = IC.jb
        je  = IC.je
        jed = IC.jed

        # get dx, dy, dz
        nx = IC.global_npx - 1
        ny = IC.global_npy - 1
        nz = npz

        domain_size = 20000
        dx = domain_size/nx
        dy = domain_size/ny
        dz = domain_size/nz

        IC.dx = dx
        IC.dy = dy
        IC.dz = dz

        if IC.global_npx == 2:
            global_x = (np.arange(1)+0.5) * dx
            IC.global_x = global_x
            x = global_x
            IC.x = x

        else:
            global_x = (np.arange(IC.global_npx+ng*2-1) + 0.5-IC.ib) * dx
            IC.global_x = global_x[ng:-ng]
            x = global_x[IC.global_ibd:IC.global_ied]
            IC.x = x[ng:-ng]

        if IC.global_npy == 2:
            global_y = (np.arange(1)+0.5) * dy
            IC.global_y = global_y
            y = global_y
            IC.y = y

        else:
            global_y = (np.arange(IC.global_npy+ng*2-1) + 0.5-IC.jb) * dy
            IC.global_y = global_y[ng:-ng]
            y = global_y[IC.global_jbd:IC.global_jed]
            IC.y = y[ng:-ng]

        z = 20000 - (np.arange(nz) + 0.5) * dz
        IC.z = z

        a_x = np.expand_dims(x[:], axis=0)
        a_dx = dx
        a_rdx = 1./a_dx

        a_y = np.expand_dims(y[:], axis=1)
        a_dy = dy
        a_rdy = 1./a_dy

        a_da = a_dx*a_dy
        a_rda = 1./a_da

        d_dx = dx
        c_dy = dy

        IC.a_x = a_x
        IC.a_dx = a_dx
        IC.a_rdx = a_rdx

        IC.a_y = a_y
        IC.a_dy = a_dy
        IC.a_rdy = a_rdy

        IC.a_da = a_da
        IC.a_rda = a_rda

        IC.d_dx = d_dx
        IC.c_dy = c_dy

        #--- progs
        delp = IC.delp
        delz = IC.delz
        ua = IC.ua
        va = IC.va
        wa = IC.wa
        pt = IC.pt

        #--- aux
        ps = IC.ps
        phis = IC.phis
        ptop = 0

        delz[:] = -dz

        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        IC.xx = xx
        IC.yy = yy
        IC.zz = zz

        IC.ptop = 0

        #--- delp, ua, pt
        pt[:] = 0

        d = 10
        wave_center = 10000
        wave_radius = 2000
        delp[0,:,:] = d*GRAV
        delp[:,:,:] += (np.sqrt((xx-wave_center)**2 + (yy-wave_center)**2)<=wave_radius)*d*GRAV*0.1

        delz[:] = -delp/GRAV # assume rho=1
        ua[:] = 0
        va[:] = 0

        return
    
    @staticmethod
    def robert_gauss_bubble(IC: ICdata):

        #--- dims
        npx = IC.npx
        npy = IC.npy
        npz = IC.npz
        ng  = IC.ng

        ibd = IC.ibd
        ib  = IC.ib
        ie  = IC.ie
        ied = IC.ied

        jbd = IC.jbd
        jb  = IC.jb
        je  = IC.je
        jed = IC.jed

        # get dx, dy, dz
        nx = IC.global_npx - 1
        ny = IC.global_npy - 1
        nz = npz
        dx = 1000/nx
        dy = 1000/ny
        dz = 1500/nz

        IC.dx = dx
        IC.dy = dy
        IC.dz = dz
        
        if IC.global_npx == 2:
            global_x = (np.arange(1)+0.5) * dx
            IC.global_x = global_x
            x = global_x
            IC.x = x

        else:
            global_x = (np.arange(IC.global_npx+ng*2-1) + 0.5-IC.ib) * dx
            IC.global_x = global_x[ng:-ng]
            x = global_x[IC.global_ibd:IC.global_ied]
            IC.x = x[ng:-ng]

        if IC.global_npy == 2:
            global_y = (np.arange(1)+0.5) * dy
            IC.global_y = global_y
            y = global_y
            IC.y = y

        else:
            global_y = (np.arange(IC.global_npy+ng*2-1) + 0.5-IC.jb) * dy
            IC.global_y = global_y[ng:-ng]
            y = global_y[IC.global_jbd:IC.global_jed]
            IC.y = y[ng:-ng]

        z = 1500 - (np.arange(nz) + 0.5) * dz
        IC.z = z

        a_x = np.expand_dims(x[:], axis=0)
        a_dx = dx
        a_rdx = 1./a_dx

        a_y = np.expand_dims(y[:], axis=1)
        a_dy = dy
        a_rdy = 1./a_dy

        a_da = a_dx*a_dy
        a_rda = 1./a_da

        d_dx = dx
        c_dy = dy

        IC.a_x = a_x
        IC.a_dx = a_dx
        IC.a_rdx = a_rdx

        IC.a_y = a_y
        IC.a_dy = a_dy
        IC.a_rdy = a_rdy

        IC.a_da = a_da
        IC.a_rda = a_rda

        IC.d_dx = d_dx
        IC.c_dy = c_dy

        #--- progs
        delp = IC.delp
        delz = IC.delz
        ua = IC.ua
        va = IC.va
        wa = IC.wa
        pt = IC.pt

        #--- aux
        ps = IC.ps
        phis = IC.phis
        ptop = 0

        delz[:] = -dz

        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        IC.xx = xx
        IC.yy = yy
        IC.zz = zz

        #--- pressure - use pkappa relation to stack up
        cp_air = CP_AIR
        kappa = KAPPA
        grav = GRAV

        t00 = 303.15
        p00 = 1.e5
        pk0 = p00**kappa

        ps[:] = p00

        pk = IC.pk
        dpk = IC.dpk

        dpk0 = grav*dz*pk0/(t00*cp_air)

        pk[:] = pk0
        dpk[:] = dpk0

        dpk = np.cumsum(dpk, axis=0)
        dpk = np.flip(dpk, axis=0)

        pk[:-1,:,:] -= dpk
        pe = pk**(1/kappa)

        ptop = pe[0,0,0]

        delp[:] = np.diff(pe, axis=0)

        # temperature - potential temperature
        pt[:] = t00

        A = 0.5
        x0 = 500
        y0 = 500
        z0 = 260
        a = 50
        s = 100

        r = np.sqrt((xx-x0)**2+(yy-y0)**2+(zz-z0)**2)

        pt[:,:,:] += A * ( r <= a ) + A * np.exp( - (r-a)**2/s**2 ) * ( r > a )

        IC.ptop = ptop

        return
    
    @staticmethod
    def robert_gauss_bubble_real(IC: ICdata):

        #--- dims
        npx = IC.npx
        npy = IC.npy
        npz = IC.npz
        ng  = IC.ng

        ibd = IC.ibd
        ib  = IC.ib
        ie  = IC.ie
        ied = IC.ied

        jbd = IC.jbd
        jb  = IC.jb
        je  = IC.je
        jed = IC.jed

        # get dx, dy, dz
        nx = IC.global_npx - 1
        ny = IC.global_npy - 1
        nz = npz
        dx = 100000/nx
        dy = 100000/ny
        dz = 1000/nz

        IC.dx = dx
        IC.dy = dy
        IC.dz = dz
        
        if IC.global_npx == 2:
            global_x = (np.arange(1)+0.5) * dx
            IC.global_x = global_x
            x = global_x
            IC.x = x

        else:
            global_x = (np.arange(IC.global_npx+ng*2-1) + 0.5-IC.ib) * dx
            IC.global_x = global_x[ng:-ng]
            x = global_x[IC.global_ibd:IC.global_ied]
            IC.x = x[ng:-ng]

        if IC.global_npy == 2:
            global_y = (np.arange(1)+0.5) * dy
            IC.global_y = global_y
            y = global_y
            IC.y = y

        else:
            global_y = (np.arange(IC.global_npy+ng*2-1) + 0.5-IC.jb) * dy
            IC.global_y = global_y[ng:-ng]
            y = global_y[IC.global_jbd:IC.global_jed]
            IC.y = y[ng:-ng]

        z = 1000 - (np.arange(nz) + 0.5) * dz
        IC.z = z

        a_x = np.expand_dims(x[:], axis=0)
        a_dx = dx
        a_rdx = 1./a_dx

        a_y = np.expand_dims(y[:], axis=1)
        a_dy = dy
        a_rdy = 1./a_dy

        a_da = a_dx*a_dy
        a_rda = 1./a_da

        d_dx = dx
        c_dy = dy

        IC.a_x = a_x
        IC.a_dx = a_dx
        IC.a_rdx = a_rdx

        IC.a_y = a_y
        IC.a_dy = a_dy
        IC.a_rdy = a_rdy

        IC.a_da = a_da
        IC.a_rda = a_rda

        IC.d_dx = d_dx
        IC.c_dy = c_dy

        #--- progs
        delp = IC.delp
        delz = IC.delz
        ua = IC.ua
        va = IC.va
        wa = IC.wa
        pt = IC.pt

        #--- aux
        ps = IC.ps
        phis = IC.phis
        ptop = 0

        delz[:] = -dz

        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        IC.xx = xx
        IC.yy = yy
        IC.zz = zz

        #--- pressure - use pkappa relation to stack up
        cp_air = CP_AIR
        kappa = KAPPA
        grav = GRAV

        t00 = 303.15
        p00 = 1.e5
        pk0 = p00**kappa

        ps[:] = p00

        pk = IC.pk
        dpk = IC.dpk

        dpk0 = grav*dz*pk0/(t00*cp_air)

        pk[:] = pk0
        dpk[:] = dpk0

        dpk = np.cumsum(dpk, axis=0)
        dpk = np.flip(dpk, axis=0)

        pk[:-1,:,:] -= dpk
        pe = pk**(1/kappa)

        ptop = pe[0,0,0]

        delp[:] = np.diff(pe, axis=0)

        # temperature - potential temperature
        pt[:] = t00

        A = 0.5
        x0 = 50000
        y0 = 50000
        z0 = 260
        a = 50
        s = 100

        r = np.sqrt(((xx-x0)/100)**2+((yy-y0)/100)**2+(zz-z0)**2)

        pt[:,:,:] += A * ( r <= a ) + A * np.exp( - (r-a)**2/s**2 ) * ( r > a )

        IC.ptop = ptop

        return
    
    @staticmethod
    def robert_uniform_bubble(IC: ICdata):

        #--- dims
        npx = IC.npx
        npy = IC.npy
        npz = IC.npz
        ng  = IC.ng

        ibd = IC.ibd
        ib  = IC.ib
        ie  = IC.ie
        ied = IC.ied

        jbd = IC.jbd
        jb  = IC.jb
        je  = IC.je
        jed = IC.jed

        # get dx, dy, dz
        nx = IC.global_npx - 1
        ny = IC.global_npy - 1
        nz = npz
        dx = 1000/nx
        dy = 1000/ny
        dz = 1000/nz

        IC.dx = dx
        IC.dy = dy
        IC.dz = dz
        
        if IC.global_npx == 2:
            global_x = (np.arange(1)+0.5) * dx
            IC.global_x = global_x
            x = global_x
            IC.x = x

        else:
            global_x = (np.arange(IC.global_npx+ng*2-1) + 0.5-IC.ib) * dx
            IC.global_x = global_x[ng:-ng]
            x = global_x[IC.global_ibd:IC.global_ied]
            IC.x = x[ng:-ng]

        if IC.global_npy == 2:
            global_y = (np.arange(1)+0.5) * dy
            IC.global_y = global_y
            y = global_y
            IC.y = y

        else:
            global_y = (np.arange(IC.global_npy+ng*2-1) + 0.5-IC.jb) * dy
            IC.global_y = global_y[ng:-ng]
            y = global_y[IC.global_jbd:IC.global_jed]
            IC.y = y[ng:-ng]

        z = 1000 - (np.arange(nz) + 0.5) * dz
        IC.z = z

        a_x = np.expand_dims(x[:], axis=0)
        a_dx = dx
        a_rdx = 1./a_dx

        a_y = np.expand_dims(y[:], axis=1)
        a_dy = dy
        a_rdy = 1./a_dy

        a_da = a_dx*a_dy
        a_rda = 1./a_da

        d_dx = dx
        c_dy = dy

        IC.a_x = a_x
        IC.a_dx = a_dx
        IC.a_rdx = a_rdx

        IC.a_y = a_y
        IC.a_dy = a_dy
        IC.a_rdy = a_rdy

        IC.a_da = a_da
        IC.a_rda = a_rda

        IC.d_dx = d_dx
        IC.c_dy = c_dy

        #--- progs
        delp = IC.delp
        delz = IC.delz
        ua = IC.ua
        va = IC.va
        wa = IC.wa
        pt = IC.pt

        #--- aux
        ps = IC.ps
        phis = IC.phis
        ptop = 0

        delz[:] = -dz

        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        IC.xx = xx
        IC.yy = yy
        IC.zz = zz

        #--- pressure - use pkappa relation to stack up
        cp_air = CP_AIR
        kappa = KAPPA
        grav = GRAV

        t00 = 303.15
        p00 = 1.e5
        pk0 = p00**kappa

        ps[:] = p00

        pk = IC.pk
        dpk = IC.dpk

        dpk0 = grav*dz*pk0/(t00*cp_air)

        pk[:] = pk0
        dpk[:] = dpk0

        dpk = np.cumsum(dpk, axis=0)
        dpk = np.flip(dpk, axis=0)

        pk[:-1,:,:] -= dpk
        pe = pk**(1/kappa)

        ptop = pe[0,0,0]

        delp[:] = np.diff(pe, axis=0)

        # temperature - potential temperature
        pt[:] = t00

        A = 0.5
        x0 = 500
        y0 = 500
        z0 = 260
        a = 250

        r = np.sqrt((xx-x0)**2+(yy-y0)**2+(zz-z0)**2)

        pt[:,:,:] += A * ( r <= a )

        IC.ptop = ptop

        return
    
    @staticmethod
    def acoustic_wave_damping(IC: ICdata):

        #--- dims
        npx = IC.npx
        npy = IC.npy
        npz = IC.npz
        ng  = IC.ng

        ibd = IC.ibd
        ib  = IC.ib
        ie  = IC.ie
        ied = IC.ied

        jbd = IC.jbd
        jb  = IC.jb
        je  = IC.je
        jed = IC.jed

        # get dx, dy, dz
        nx = IC.global_npx - 1
        ny = IC.global_npy - 1
        nz = npz
        dx = 1000/nx
        dy = 1000/ny
        dz = 680/nz

        IC.dx = dx
        IC.dy = dy
        IC.dz = dz
        
        if IC.global_npx == 2:
            global_x = (np.arange(1)+0.5) * dx
            IC.global_x = global_x
            x = global_x
            IC.x = x

        else:
            global_x = (np.arange(IC.global_npx+ng*2-1) + 0.5-IC.ib) * dx
            IC.global_x = global_x[ng:-ng]
            x = global_x[IC.global_ibd:IC.global_ied]
            IC.x = x[ng:-ng]

        if IC.global_npy == 2:
            global_y = (np.arange(1)+0.5) * dy
            IC.global_y = global_y
            y = global_y
            IC.y = y

        else:
            global_y = (np.arange(IC.global_npy+ng*2-1) + 0.5-IC.jb) * dy
            IC.global_y = global_y[ng:-ng]
            y = global_y[IC.global_jbd:IC.global_jed]
            IC.y = y[ng:-ng]

        z = 680 - (np.arange(nz) + 0.5) * dz
        IC.z = z

        a_x = np.expand_dims(x[:], axis=0)
        a_dx = dx
        a_rdx = 1./a_dx

        a_y = np.expand_dims(y[:], axis=1)
        a_dy = dy
        a_rdy = 1./a_dy

        a_da = a_dx*a_dy
        a_rda = 1./a_da

        d_dx = dx
        c_dy = dy

        IC.a_x = a_x
        IC.a_dx = a_dx
        IC.a_rdx = a_rdx

        IC.a_y = a_y
        IC.a_dy = a_dy
        IC.a_rdy = a_rdy

        IC.a_da = a_da
        IC.a_rda = a_rda

        IC.d_dx = d_dx
        IC.c_dy = c_dy

        #--- progs
        delp = IC.delp
        delz = IC.delz
        ua = IC.ua
        va = IC.va
        wa = IC.wa
        pt = IC.pt

        #--- aux
        ps = IC.ps
        phis = IC.phis
        ptop = 0

        delz[:] = -dz

        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        IC.xx = xx
        IC.yy = yy
        IC.zz = zz

        #--- pressure - use pkappa relation to stack up
        cp_air = CP_AIR
        kappa = KAPPA
        grav = GRAV

        t00 = 290
        p00 = 1.e5
        pk0 = p00**kappa

        ps[:] = p00

        pk = IC.pk
        dpk = IC.dpk

        dpk0 = grav*dz*pk0/(t00*cp_air)

        pk[:] = pk0
        dpk[:] = dpk0

        dpk = np.cumsum(dpk, axis=0)
        dpk = np.flip(dpk, axis=0)

        pk[:-1,:,:] -= dpk
        pe = pk**(1/kappa)

        ptop = pe[0,0,0]

        delp[:] = np.diff(pe, axis=0)

        # temperature - potential temperature
        pt[:] = t00

        A = 0.1
        z0 = 340

        r = (zz-z0)/340*np.pi

        pt[:,:,:] += A * np.sin(r)

        IC.ptop = ptop

        return
    
    @staticmethod
    def gravity_internal_wave(IC: ICdata):

        #--- dims
        npx = IC.npx
        npy = IC.npy
        npz = IC.npz
        ng  = IC.ng

        ibd = IC.ibd
        ib  = IC.ib
        ie  = IC.ie
        ied = IC.ied

        jbd = IC.jbd
        jb  = IC.jb
        je  = IC.je
        jed = IC.jed

        # get dx, dy, dz
        nx = IC.global_npx - 1
        ny = IC.global_npy - 1
        nz = npz

        domain_x = 300000
        domain_y = 300000
        domain_z = 10000
        dx = domain_x/nx
        dy = domain_y/ny
        dz = domain_z/nz

        IC.dx = dx
        IC.dy = dy
        IC.dz = dz
        
        if IC.global_npx == 2:
            global_x = (np.arange(1)+0.5) * dx
            IC.global_x = global_x
            x = global_x
            IC.x = x

        else:
            global_x = (np.arange(IC.global_npx+ng*2-1) + 0.5-IC.ib) * dx
            IC.global_x = global_x[ng:-ng]
            x = global_x[IC.global_ibd:IC.global_ied]
            IC.x = x[ng:-ng]

        if IC.global_npy == 2:
            global_y = (np.arange(1)+0.5) * dy
            IC.global_y = global_y
            y = global_y
            IC.y = y

        else:
            global_y = (np.arange(IC.global_npy+ng*2-1) + 0.5-IC.jb) * dy
            IC.global_y = global_y[ng:-ng]
            y = global_y[IC.global_jbd:IC.global_jed]
            IC.y = y[ng:-ng]

        z = domain_z - (np.arange(nz) + 0.5) * dz
        IC.z = z

        a_x = np.expand_dims(x[:], axis=0)
        a_dx = dx
        a_rdx = 1./a_dx

        a_y = np.expand_dims(y[:], axis=1)
        a_dy = dy
        a_rdy = 1./a_dy

        a_da = a_dx*a_dy
        a_rda = 1./a_da

        d_dx = dx
        c_dy = dy

        IC.a_x = a_x
        IC.a_dx = a_dx
        IC.a_rdx = a_rdx

        IC.a_y = a_y
        IC.a_dy = a_dy
        IC.a_rdy = a_rdy

        IC.a_da = a_da
        IC.a_rda = a_rda

        IC.d_dx = d_dx
        IC.c_dy = c_dy

        #--- progs
        delp = IC.delp
        delz = IC.delz
        ua = IC.ua
        va = IC.va
        wa = IC.wa
        pt = IC.pt

        #--- aux
        ps = IC.ps
        phis = IC.phis
        ptop = 0

        delz[:] = -dz

        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        IC.xx = xx
        IC.yy = yy
        IC.zz = zz

        #--- pressure - use pkappa relation to stack up
        cp_air = CP_AIR
        kappa = KAPPA
        grav = GRAV

        t00 = 300.
        p00 = 1.e5

        N = 0.01
        
        pt[:] = t00*np.exp(N**2/grav*(zz-dz))

        pi_ave = 1 + grav**2/(cp_air*t00*N**2)*(np.exp(-N**2/grav*(zz-dz))-1)

        pr = pi_ave**(1./kappa)*p00

        ptop = pr[0,0,0]

        t = pt * pi_ave

        rho = pr/RDGAS/t

        delp[:] = grav*dz*rho
        
        A = 0.01
        xc = 90000
        #yc = 150000
        a = 5000

        #pt[:,:,:] += A * (np.sin(np.pi*zz/domain_z))/(1+((xx-xc)**2+(yy-yc)**2)/a**2)
        pt[:,:,:] += A * (np.sin(np.pi*zz/domain_z))/(1+((xx-xc)**2)/a**2)
        
        ua[:] = 20
        #va[:] = 20
        
        IC.ptop = ptop
        ps[:] = p00

        return
    
    @staticmethod
    def gravity_internal_wave_asymmetric(IC: ICdata):

        #--- dims
        npx = IC.npx
        npy = IC.npy
        npz = IC.npz
        ng  = IC.ng

        ibd = IC.ibd
        ib  = IC.ib
        ie  = IC.ie
        ied = IC.ied

        jbd = IC.jbd
        jb  = IC.jb
        je  = IC.je
        jed = IC.jed

        # get dx, dy, dz
        nx = IC.global_npx - 1
        ny = IC.global_npy - 1
        nz = npz

        domain_x = 300000
        domain_y = 300000
        domain_z = 10000
        dx = domain_x/nx
        dy = domain_y/ny
        dz = domain_z/nz

        IC.dx = dx
        IC.dy = dy
        IC.dz = dz
        
        if IC.global_npx == 2:
            global_x = (np.arange(1)+0.5) * dx
            IC.global_x = global_x
            x = global_x
            IC.x = x

        else:
            global_x = (np.arange(IC.global_npx+ng*2-1) + 0.5-IC.ib) * dx
            IC.global_x = global_x[ng:-ng]
            x = global_x[IC.global_ibd:IC.global_ied]
            IC.x = x[ng:-ng]

        if IC.global_npy == 2:
            global_y = (np.arange(1)+0.5) * dy
            IC.global_y = global_y
            y = global_y
            IC.y = y

        else:
            global_y = (np.arange(IC.global_npy+ng*2-1) + 0.5-IC.jb) * dy
            IC.global_y = global_y[ng:-ng]
            y = global_y[IC.global_jbd:IC.global_jed]
            IC.y = y[ng:-ng]

        z = domain_z - (np.arange(nz) + 0.5) * dz
        IC.z = z

        a_x = np.expand_dims(x[:], axis=0)
        a_dx = dx
        a_rdx = 1./a_dx

        a_y = np.expand_dims(y[:], axis=1)
        a_dy = dy
        a_rdy = 1./a_dy

        a_da = a_dx*a_dy
        a_rda = 1./a_da

        d_dx = dx
        c_dy = dy

        IC.a_x = a_x
        IC.a_dx = a_dx
        IC.a_rdx = a_rdx

        IC.a_y = a_y
        IC.a_dy = a_dy
        IC.a_rdy = a_rdy

        IC.a_da = a_da
        IC.a_rda = a_rda

        IC.d_dx = d_dx
        IC.c_dy = c_dy

        #--- progs
        delp = IC.delp
        delz = IC.delz
        ua = IC.ua
        va = IC.va
        wa = IC.wa
        pt = IC.pt

        #--- aux
        ps = IC.ps
        phis = IC.phis
        ptop = 0

        delz[:] = -dz

        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        IC.xx = xx
        IC.yy = yy
        IC.zz = zz

        #--- pressure - use pkappa relation to stack up
        cp_air = CP_AIR
        kappa = KAPPA
        grav = GRAV

        t00 = 300.
        p00 = 1.e5

        N = 0.01
        
        pt[:] = t00*np.exp(N**2/grav*(zz-dz))

        pi_ave = 1 + grav**2/(cp_air*t00*N**2)*(np.exp(-N**2/grav*(zz-dz))-1)

        pr = pi_ave**(1./kappa)*p00

        ptop = pr[0,0,0]

        t = pt * pi_ave

        rho = pr/RDGAS/t

        delp[:] = grav*dz*rho
        
        A = 0.01
        xc = 150000
        yc = 150000
        a = 5000

        pt[:,:,:] += A * (np.sin(np.pi*zz/domain_z))/(1+((xx-xc)**2+(yy-yc)**2)/a**2)
        #pt[:,:,:] += A * (np.sin(np.pi*zz/domain_z))/(1+((xx-xc)**2)/a**2)
        
        #ua[:] = 20
        #va[:] = 20
        
        IC.ptop = ptop
        ps[:] = p00

        return
    
    @staticmethod
    def straka_sinking_bubble(IC: ICdata):

        #--- dims
        npx = IC.npx
        npy = IC.npy
        npz = IC.npz
        ng  = IC.ng

        ibd = IC.ibd
        ib  = IC.ib
        ie  = IC.ie
        ied = IC.ied

        jbd = IC.jbd
        jb  = IC.jb
        je  = IC.je
        jed = IC.jed

        # get dx, dy, dz
        nx = IC.global_npx - 1
        ny = IC.global_npy - 1
        nz = npz
        dx = 36000/nx
        dy = 36000/ny
        dz = 6000/nz

        IC.dx = dx
        IC.dy = dy
        IC.dz = dz
        
        if IC.global_npx == 2:
            global_x = (np.arange(1)+0.5) * dx
            IC.global_x = global_x
            x = global_x
            IC.x = x

        else:
            global_x = (np.arange(IC.global_npx+ng*2-1) + 0.5-IC.ib) * dx
            IC.global_x = global_x[ng:-ng]
            x = global_x[IC.global_ibd:IC.global_ied]
            IC.x = x[ng:-ng]

        if IC.global_npy == 2:
            global_y = (np.arange(1)+0.5) * dy
            IC.global_y = global_y
            y = global_y
            IC.y = y

        else:
            global_y = (np.arange(IC.global_npy+ng*2-1) + 0.5-IC.jb) * dy
            IC.global_y = global_y[ng:-ng]
            y = global_y[IC.global_jbd:IC.global_jed]
            IC.y = y[ng:-ng]

        z = 6000 - (np.arange(nz) + 0.5) * dz
        IC.z = z

        a_x = np.expand_dims(x[:], axis=0)
        a_dx = dx
        a_rdx = 1./a_dx

        a_y = np.expand_dims(y[:], axis=1)
        a_dy = dy
        a_rdy = 1./a_dy

        a_da = a_dx*a_dy
        a_rda = 1./a_da

        d_dx = dx
        c_dy = dy

        IC.a_x = a_x
        IC.a_dx = a_dx
        IC.a_rdx = a_rdx

        IC.a_y = a_y
        IC.a_dy = a_dy
        IC.a_rdy = a_rdy

        IC.a_da = a_da
        IC.a_rda = a_rda

        IC.d_dx = d_dx
        IC.c_dy = c_dy

        #--- progs
        delp = IC.delp
        delz = IC.delz
        ua = IC.ua
        va = IC.va
        wa = IC.wa
        pt = IC.pt

        #--- aux
        ps = IC.ps
        phis = IC.phis
        ptop = 0

        delz[:] = -dz

        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        IC.xx = xx
        IC.yy = yy
        IC.zz = zz

        #--- pressure - use pkappa relation to stack up
        cp_air = CP_AIR
        kappa = KAPPA
        grav = GRAV

        t00 = 300
        p00 = 1.e5
        pk0 = p00**kappa

        ps[:] = p00

        pk = IC.pk
        dpk = IC.dpk

        dpk0 = grav*dz*pk0/(t00*cp_air)

        pk[:] = pk0
        dpk[:] = dpk0

        dpk = np.cumsum(dpk, axis=0)
        dpk = np.flip(dpk, axis=0)

        pk[:-1,:,:] -= dpk
        pe = pk**(1/kappa)

        ptop = pe[0,0,0]

        delp[:] = np.diff(pe, axis=0)

        # temperature - potential temperature
        pt[:] = t00

        A = -15
        x0 = 18000
        y0 = 18000
        z0 = 3000
        
        xr = 4000
        yr = 4000
        zr = 2000

        r = np.sqrt(((xx-x0)/xr)**2+((yy-y0)/yr)**2+((zz-z0)/zr)**2)

        pt[:,:,:] += A * ( r <= 1 )*(np.cos(np.pi*r)+1)/2

        IC.ptop = ptop

        return
    
    @staticmethod
    def straka_sinking_uniform_bubble(IC: ICdata):

        #--- dims
        npx = IC.npx
        npy = IC.npy
        npz = IC.npz
        ng  = IC.ng

        ibd = IC.ibd
        ib  = IC.ib
        ie  = IC.ie
        ied = IC.ied

        jbd = IC.jbd
        jb  = IC.jb
        je  = IC.je
        jed = IC.jed

        # get dx, dy, dz
        nx = IC.global_npx - 1
        ny = IC.global_npy - 1
        nz = npz
        dx = 36000/nx
        dy = 36000/ny
        dz = 6000/nz

        IC.dx = dx
        IC.dy = dy
        IC.dz = dz
        
        if IC.global_npx == 2:
            global_x = (np.arange(1)+0.5) * dx
            IC.global_x = global_x
            x = global_x
            IC.x = x

        else:
            global_x = (np.arange(IC.global_npx+ng*2-1) + 0.5-IC.ib) * dx
            IC.global_x = global_x[ng:-ng]
            x = global_x[IC.global_ibd:IC.global_ied]
            IC.x = x[ng:-ng]

        if IC.global_npy == 2:
            global_y = (np.arange(1)+0.5) * dy
            IC.global_y = global_y
            y = global_y
            IC.y = y

        else:
            global_y = (np.arange(IC.global_npy+ng*2-1) + 0.5-IC.jb) * dy
            IC.global_y = global_y[ng:-ng]
            y = global_y[IC.global_jbd:IC.global_jed]
            IC.y = y[ng:-ng]

        z = 6000 - (np.arange(nz) + 0.5) * dz
        IC.z = z

        a_x = np.expand_dims(x[:], axis=0)
        a_dx = dx
        a_rdx = 1./a_dx

        a_y = np.expand_dims(y[:], axis=1)
        a_dy = dy
        a_rdy = 1./a_dy

        a_da = a_dx*a_dy
        a_rda = 1./a_da

        d_dx = dx
        c_dy = dy

        IC.a_x = a_x
        IC.a_dx = a_dx
        IC.a_rdx = a_rdx

        IC.a_y = a_y
        IC.a_dy = a_dy
        IC.a_rdy = a_rdy

        IC.a_da = a_da
        IC.a_rda = a_rda

        IC.d_dx = d_dx
        IC.c_dy = c_dy

        #--- progs
        delp = IC.delp
        delz = IC.delz
        ua = IC.ua
        va = IC.va
        wa = IC.wa
        pt = IC.pt

        #--- aux
        ps = IC.ps
        phis = IC.phis
        ptop = 0

        delz[:] = -dz

        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        IC.xx = xx
        IC.yy = yy
        IC.zz = zz

        #--- pressure - use pkappa relation to stack up
        cp_air = CP_AIR
        kappa = KAPPA
        grav = GRAV

        t00 = 300
        p00 = 1.e5
        pk0 = p00**kappa

        ps[:] = p00

        pk = IC.pk
        dpk = IC.dpk

        dpk0 = grav*dz*pk0/(t00*cp_air)

        pk[:] = pk0
        dpk[:] = dpk0

        dpk = np.cumsum(dpk, axis=0)
        dpk = np.flip(dpk, axis=0)

        pk[:-1,:,:] -= dpk
        pe = pk**(1/kappa)

        ptop = pe[0,0,0]

        delp[:] = np.diff(pe, axis=0)

        # temperature - potential temperature
        pt[:] = t00

        A = -15
        x0 = 18000
        y0 = 18000
        z0 = 3000
        
        xr = 4000
        yr = 4000
        zr = 2000

        r = np.sqrt(((xx-x0)/xr)**2+((yy-y0)/yr)**2+((zz-z0)/zr)**2)

        pt[:,:,:] += A * ( r <= 0.6 )

        IC.ptop = ptop

        return
    
