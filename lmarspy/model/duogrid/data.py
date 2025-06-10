# Purpose: Duogrid data type

from lmarspy.backend.field import Field
from lmarspy.backend.cal import cal
from lmarspy.model.mp import Mp

class DuogridData:

    def __init__(self, mp: Mp) -> None:

        self.mp = mp

        npx = mp.npx; npy = mp.npy; npz = mp.npz; ng  = mp.ng
        ibd = mp.ibd; ib  = mp.ib;  ie  = mp.ie;  ied = mp.ied
        jbd = mp.jbd; jb  = mp.jb;  je  = mp.je;  jed = mp.jed
        
        #--- assign a-pt
        self.a_pt = Field(dims = mp.dim_ha+[2])

        self.a_x = Field(dims = mp.dim_ha)
        self.a_y = Field(dims = mp.dim_ha)

        self.a_kik_x = Field(dims = mp.dim_ha)
        self.a_kik_y = Field(dims = mp.dim_ha)

        self.a_gco = Field(dims = mp.dim_ha+[2, 2])
        self.a_gct = Field(dims = mp.dim_ha+[2, 2])

        self.a_c2l = Field(dims = mp.dim_ha+[2, 2])
        self.a_l2c = Field(dims = mp.dim_ha+[2, 2])

        self.a_sina = Field(dims = mp.dim_ha)
        self.a_cosa = Field(dims = mp.dim_ha)

        self.a_dx = Field(dims = mp.dim_ha)
        self.a_dy = Field(dims = mp.dim_ha)
        self.a_da  = Field(dims = mp.dim_ha)

        self.a_rdx = Field(dims = mp.dim_ha)
        self.a_rdy = Field(dims = mp.dim_ha)
        self.a_rda = Field(dims = mp.dim_ha)

        #--- assign b-pt
        self.b_pt = Field(dims = mp.dim_hb+[2])

        #--- assign c-pt
        self.c_dy = Field(dims = mp.dim_hc)

        self.c_gco = Field(dims = mp.dim_hc+[2, 2])
        self.c_gct = Field(dims = mp.dim_hc+[2, 2])

        self.c_ct2ort_x = Field(dims = mp.dim_hc+[2, 2])
        self.c_ort2ct_x = Field(dims = mp.dim_hc+[2, 2])

        self.c_sina = Field(dims = mp.dim_hc)
        self.c_cosa = Field(dims = mp.dim_hc)

        #--- assign d-pt
        self.d_dx = Field(dims = mp.dim_hd)

        self.d_gco = Field(dims = mp.dim_hd+[2, 2])
        self.d_gct = Field(dims = mp.dim_hd+[2, 2])

        self.d_ct2ort_y = Field(dims = mp.dim_hd+[2, 2])
        self.d_ort2ct_y = Field(dims = mp.dim_hd+[2, 2])

        self.d_sina = Field(dims = mp.dim_hd)
        self.d_cosa = Field(dims = mp.dim_hd)

        return


