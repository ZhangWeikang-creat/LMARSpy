from lmarspy.backend.field import Field
from lmarspy.backend.cal import cal
from lmarspy.model.mp import Mp

from .data import DuogridData


class DuogridUtil:

    @staticmethod
    def assign_cart_metrics(dg: DuogridData) -> None:
        '''TODO: 还有其他 2x2 的几何项需要处理'''
        '''TODO: 测试a_gct'''

        mp = dg.mp

        npx = mp.npx; npy = mp.npy; npz = mp.npz; ng  = mp.ng
        ibd = mp.ibd; ib  = mp.ib;  ie  = mp.ie;  ied = mp.ied
        jbd = mp.jbd; jb  = mp.jb;  je  = mp.je;  jed = mp.jed

        dg.a_sina[:] = Field(data=1.)
        dg.a_cosa[:] = Field(data=0.)
        
        dg.c_sina[:] = Field(data=1.)
        dg.c_cosa[:] = Field(data=0.)

        dg.d_sina[:] = Field(data=1.)
        dg.d_cosa[:] = Field(data=0.)

        # 2x2 metrics of a-pt
        dg.a_gct[:] = cal.unsqueeze(
                cal.unsqueeze(
                    Field(data=[[1,0],[0,1]]), dim=0),
                dim=0)

        dg.a_gco[:] = cal.unsqueeze(
                cal.unsqueeze(
                    Field(data=[[1,0],[0,1]]), dim=0),
                dim=0)

        dg.a_c2l[:] = cal.unsqueeze(
                cal.unsqueeze(
                    Field(data=[[1,0],[0,1]]), dim=0),
                dim=0)

        dg.a_l2c[:] = cal.unsqueeze(
                cal.unsqueeze(
                    Field(data=[[1,0],[0,1]]), dim=0),
                dim=0)

        # 2x2 metrics of c-pt
        dg.c_gct[:] = cal.unsqueeze(
                cal.unsqueeze(
                    Field(data=[[1,0],[0,1]]), dim=0),
                dim=0)

        dg.c_gco[:] = cal.unsqueeze(
                cal.unsqueeze(
                    Field(data=[[1,0],[0,1]]), dim=0),
                dim=0)

        dg.c_ct2ort_x[:] = cal.unsqueeze(
                cal.unsqueeze(
                    Field(data=[[1,0],[0,1]]), dim=0),
                dim=0)
        
        dg.c_ort2ct_x[:] = cal.unsqueeze(
                cal.unsqueeze(
                    Field(data=[[1,0],[0,1]]), dim=0),
                dim=0)
        
        # 2x2 metrics of d-pt
        dg.d_gct[:] = cal.unsqueeze(
                cal.unsqueeze(
                    Field(data=[[1,0],[0,1]]), dim=0),
                dim=0)

        dg.d_gco[:] = cal.unsqueeze(
                cal.unsqueeze(
                    Field(data=[[1,0],[0,1]]), dim=0),
                dim=0)
        
        dg.d_ct2ort_y[:] = cal.unsqueeze(
                cal.unsqueeze(
                    Field(data=[[1,0],[0,1]]), dim=0),
                dim=0)
        
        dg.d_ort2ct_y[:] = cal.unsqueeze(
                cal.unsqueeze(
                    Field(data=[[1,0],[0,1]]), dim=0),
                dim=0)

        return


    @staticmethod
    def ext_scalar(var: Field, mp:Mp):
        #--- dim parameters
        ng = mp.ng
        npx = mp.npx
        npy = mp.npy
        npz = mp.npz

        ib  = mp.ib
        ie  = mp.ie
        ibd = mp.ibd
        ied = mp.ied

        jb  = mp.jb
        je  = mp.je
        jbd = mp.jbd
        jed = mp.jed

        if not npx == 2:
            var[:, :, ibd:ib] = var[:, :, ie-ng:ie]
            var[:, :, ie:ied] = var[:, :, ib:ib+ng]

        if not npy == 2:
            var[:, jbd:jb, :] = var[:, je-ng:je, :]
            var[:, je:jed, :] = var[:, jb:jb+ng, :]

        return
    
    @staticmethod
    def ext_vector(ua: Field, va: Field, mp:Mp):
        #--- dim parameters
        ng = mp.ng
        npx = mp.npx
        npy = mp.npy
        npz = mp.npz

        ib  = mp.ib
        ie  = mp.ie
        ibd = mp.ibd
        ied = mp.ied

        jb  = mp.jb
        je  = mp.je
        jbd = mp.jbd
        jed = mp.jed

        if not npx == 2:
            ua[:, :, ibd:ib] = ua[:, :, ie-ng:ie]
            ua[:, :, ie:ied] = ua[:, :, ib:ib+ng]

        if not npy == 2:
            ua[:, jbd:jb, :] = ua[:, je-ng:je, :]
            ua[:, je:jed, :] = ua[:, jb:jb+ng, :]

        if not npx == 2:
            va[:, :, ibd:ib] = va[:, :, ie-ng:ie]
            va[:, :, ie:ied] = va[:, :, ib:ib+ng]

        if not npy == 2:
            va[:, jbd:jb, :] = va[:, je-ng:je, :]
            va[:, je:jed, :] = va[:, jb:jb+ng, :]

        return


