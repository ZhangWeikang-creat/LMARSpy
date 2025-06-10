from lmarspy.model.atm import Atm
from lmarspy.model.sci import Sci
from .case import ICcase
from .data import ICdata

from lmarspy.backend.field import Field
from lmarspy.backend.cal import cal

from lmarspy.fps.data import FpsData

class ICbase:
    
    @staticmethod
    def assign_atm(IC: ICdata, fps:FpsData, atm: Atm):
        
        if atm.fg.ic_type == 11:
            fps.main_print('#--- Initialize atm with shallow water gauss wave test...')
            ICcase.sw_gauss_wave(IC)

        elif atm.fg.ic_type == 12:
            fps.main_print('#--- Initialize atm with shallow water square wave test...')
            ICcase.sw_square_wave(IC)

        elif atm.fg.ic_type == 21:
            fps.main_print('#--- Initialize atm with Robert 1994 warmbubble(Gaussian) test...')
            ICcase.robert_gauss_bubble(IC)

        elif atm.fg.ic_type == 22:
            fps.main_print('#--- Initialize atm with Robert 1994 warmbubble(uniform) test...')
            ICcase.robert_uniform_bubble(IC)

        elif atm.fg.ic_type == 23:
            fps.main_print('#--- Initialize atm with Robert 1994 warmbubble(Gaussian) real test...')
            ICcase.robert_gauss_bubble_real(IC)

        elif atm.fg.ic_type == 31:
            fps.main_print('#--- Initialize atm with acoustic wave damping test...')
            ICcase.acoustic_wave_damping(IC)

        elif atm.fg.ic_type == 41:
            fps.main_print('#--- Initialize atm with Straka 1993 sinking bubble test...')
            ICcase.straka_sinking_bubble(IC)

        elif atm.fg.ic_type == 42:
            fps.main_print('#--- Initialize atm with Straka 1993 sinking uniform bubble test...')
            ICcase.straka_sinking_uniform_bubble(IC)

        elif atm.fg.ic_type == 51:
            fps.main_print('#--- Initialize atm with gravity internal wave test...')
            ICcase.gravity_internal_wave(IC)
        
        elif atm.fg.ic_type == 52:
            fps.main_print('#--- Initialize atm with gravity internal wave test (asymmetric) ...')
            ICcase.gravity_internal_wave_asymmetric(IC)

        else:
            fps.raise_error(f"unsupported ic_type:{atm.fg.ic_type}")
        
        dx = IC.dx
        dy = IC.dy
        dz = IC.dz

        fps.main_print(f'    {dx=}, {dy=}, {dz=}')
        
        #--- grid
        atm.dg.a_x[:] = Field(data = IC.a_x)
        atm.dg.a_dx[:] = Field(data = IC.a_dx)
        atm.dg.a_rdx[:] = Field(data = IC.a_rdx)

        atm.dg.a_y[:] = Field(data = IC.a_y)
        atm.dg.a_dy[:] = Field(data = IC.a_dy)
        atm.dg.a_rdy[:] = Field(data = IC.a_rdy)

        atm.dg.a_da[:] = Field(data = IC.a_da)
        atm.dg.a_rda[:] = Field(data = IC.a_rda)

        atm.dg.d_dx[:] = Field(data = IC.d_dx)
        atm.dg.c_dy[:] = Field(data = IC.c_dy)

        atm.global_x = Field(data = IC.global_x)
        atm.global_y = Field(data = IC.global_y)
        
        atm.x = Field(data = IC.x)
        atm.y = Field(data = IC.y)
        atm.z = Field(data = IC.z)

        atm.xx = Field(data = IC.xx)
        atm.yy = Field(data = IC.yy)
        atm.zz = Field(data = IC.zz)

        #--- field
        atm.delp[:] = Field(data = IC.delp)
        atm.delz[:] = Field(data = IC.delz)
        atm.ua[:] = Field(data = IC.ua)
        atm.va[:] = Field(data = IC.va)
        atm.wa[:] = Field(data = IC.wa)
        atm.pt[:] = Field(data = IC.pt)

        atm.ps[:] = Field(data = IC.ps)
        atm.phis[:] = Field(data = IC.phis)
        atm.ptop = Field(data = IC.ptop)

        atm.pt[:] = Sci.pt2t(atm.pt, atm.delp, atm.delz)
