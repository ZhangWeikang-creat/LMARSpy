# Purpose: Atm  diag module

from lmarspy.backend.field import Field
from lmarspy.backend.cal import cal
from .data import AtmData
from lmarspy.model.const import Const
class AtmDiag:

    @staticmethod
    def check_var0(atm: AtmData) -> None:

        fps = atm.fps

        global_npx = fps.global_npx
        global_npy = fps.global_npy
        npz = fps.npz

        ibd = atm.mp.ibd
        ib  = atm.mp.ib
        ie  = atm.mp.ie
        ied = atm.mp.ied

        jbd = atm.mp.jbd
        jb  = atm.mp.jb
        je  = atm.mp.je
        jed = atm.mp.jed

        fps.main_print(f"# 诊断：")
        
        dp = atm.delp[:, jb:je, ib:ie]
        dz = atm.delz[:, jb:je, ib:ie]
        pt = atm.pt[:, jb:je, ib:ie]
        ua = atm.ua[:, jb:je, ib:ie]
        va = atm.va[:, jb:je, ib:ie]
        wa = atm.wa[:, jb:je, ib:ie]

        #--- 检测nan
        if cal.isnan(dp).any():
            fps.raise_error(f"There is nan in dp")
        if cal.isnan(dz).any():
            fps.raise_error(f"There is nan in dz")
        if cal.isnan(pt).any():
            fps.raise_error(f"There is nan in pt")
        if cal.isnan(ua).any():
            fps.raise_error(f"There is nan in ua")
        if cal.isnan(va).any():
            fps.raise_error(f"There is nan in va")
        if cal.isnan(wa).any():
            fps.raise_error(f"There is nan in wa")

        dz = -dz
        rho = dp/dz/Const.GRAV
        u = ua
        v = va
        w = wa
        t = pt
        p = rho*Const.RDGAS * t

        #--- 检测总质量
        da = atm.dg.a_da[jb:je, ib:ie]
        total_mass = fps.gather_sum(cal.sum(rho*da*dz))
        atm.total_mass0 = total_mass

        #--- 检测总能量
        energy = 0.5*rho*da*dz*(u**2+v**2+w**2) + (Const.CP_AIR-Const.RDGAS)*t*rho*da*dz + Const.GRAV*rho*da*dz*atm.zz[:, jb:je, ib:ie]
        total_energy = fps.gather_sum(cal.sum(energy))
        fps.main_print("    总质量：", "%.6e"%(total_mass), "总能量：", "%.6e"%(total_energy))  
        atm.total_energy0 = total_energy

        total_elements = (global_npx-1)*(global_npy-1)*npz
        #--- 检测速度
        min_ua = fps.gather_min(u.min())
        max_ua = fps.gather_max(u.max())
        mean_ua = fps.gather_sum(cal.sum(u))/total_elements
        min_va = fps.gather_min(v.min())
        max_va = fps.gather_max(v.max())
        mean_va = fps.gather_sum(cal.sum(v))/total_elements
        min_wa = fps.gather_min(w.min())
        max_wa = fps.gather_max(w.max())
        mean_wa = fps.gather_sum(cal.sum(w))/total_elements

        #--- 检测密度气压温度
        min_rho = fps.gather_min(rho.min())
        max_rho = fps.gather_max(rho.max())
        mean_rho = fps.gather_sum(cal.sum(rho))/total_elements
        min_t = fps.gather_min(t.min())
        max_t = fps.gather_max(t.max())
        mean_t = fps.gather_sum(cal.sum(t))/total_elements
        min_p = fps.gather_min(p.min())
        max_p = fps.gather_max(p.max())
        mean_p = fps.gather_sum(cal.sum(p))/total_elements

        fps.main_print("                    Min            Max           Mean")
        fps.main_print("    rho:", "    %10.4f"%(min_rho), "    %10.4f"%(max_rho), "    %10.4f"%(mean_rho))  
        fps.main_print("    u  :", "    %10.4f"%(min_ua), "    %10.4f"%(max_ua), "    %10.4f"%(mean_ua))  
        fps.main_print("    v  :", "    %10.4f"%(min_va), "    %10.4f"%(max_va), "    %10.4f"%(mean_va))  
        fps.main_print("    w  :", "    %10.4f"%(min_wa), "    %10.4f"%(max_wa), "    %10.4f"%(mean_wa))  
        fps.main_print("    P  :", "    %10.4f"%(min_p), "    %10.4f"%(max_p), "    %10.4f"%(mean_p))  
        fps.main_print("    T  :", "    %10.4f"%(min_t), "    %10.4f"%(max_t), "    %10.4f"%(mean_t))  

        #--- 检测异常值
        if min_rho <=0 :
            fps.raise_error(f"density cannot be less than 0 kg/m^3!")
        if max_ua > 400 or min_ua < -400:
            fps.raise_error(f"ua cannot exceed the speed of sound!")
        if max_va > 400 or min_va < -400:
            fps.raise_error(f"va cannot exceed the speed of sound!")
        if max_wa > 400 or min_wa < -400:
            fps.raise_error(f"wa cannot exceed the speed of sound!")
        if min_t <0 :
            fps.raise_error(f"temperature cannot be less than 0 K!")
        if min_p <0 :
            fps.raise_error(f"pressure cannot be less than 0 Pa!")

        return
    
    @staticmethod
    def check_var(atm: AtmData) -> None:

        fps = atm.fps

        global_npx = fps.global_npx
        global_npy = fps.global_npy
        npz = fps.npz

        ibd = atm.mp.ibd
        ib  = atm.mp.ib
        ie  = atm.mp.ie
        ied = atm.mp.ied

        jbd = atm.mp.jbd
        jb  = atm.mp.jb
        je  = atm.mp.je
        jed = atm.mp.jed

        fps.main_print(f"# 诊断：")
        
        dp = atm.delp
        dz = atm.delz
        pt = atm.pt
        ua = atm.ua
        va = atm.va
        wa = atm.wa

        #--- 检测nan
        if cal.isnan(dp).any():
            fps.raise_error(f"There is nan in dp")
        if cal.isnan(dz).any():
            fps.raise_error(f"There is nan in dz")
        if cal.isnan(pt).any():
            fps.raise_error(f"There is nan in pt")
        if cal.isnan(ua).any():
            fps.raise_error(f"There is nan in ua")
        if cal.isnan(va).any():
            fps.raise_error(f"There is nan in va")
        if cal.isnan(wa).any():
            fps.raise_error(f"There is nan in wa")

        dz = -dz[:, jb:je, ib:ie]
        rho = dp[:, jb:je, ib:ie]/dz/Const.GRAV
        u = ua[:, jb:je, ib:ie]
        v = va[:, jb:je, ib:ie]
        w = wa[:, jb:je, ib:ie]
        t = pt[:, jb:je, ib:ie]
        p = rho*Const.RDGAS * t

        #--- 检测总质量
        da = atm.dg.a_da[jb:je, ib:ie]
        total_mass = fps.gather_sum(cal.sum(rho*da*dz))
        mass_change = cal.abs((total_mass-atm.total_mass0)/atm.total_mass0)
        fps.main_print("    总质量：", "%.6e"%(total_mass), "变化：", "%.4f"%(mass_change*100), "%")  

        #--- 检测总能量
        energy = 0.5*rho*da*dz*(u**2+v**2+w**2) + (Const.CP_AIR-Const.RDGAS)*t*rho*da*dz + Const.GRAV*rho*da*dz*atm.zz[:, jb:je, ib:ie]
        total_energy = fps.gather_sum(cal.sum(energy))
        energy_change = cal.abs((total_energy-atm.total_energy0)/atm.total_energy0)
        fps.main_print("    总能量：", "%.6e"%(total_energy), "变化：", "%.4f"%(energy_change*100), "%")  

        total_elements = (global_npx-1)*(global_npy-1)*npz
        #--- 检测速度
        min_ua = fps.gather_min(u.min())
        max_ua = fps.gather_max(u.max())
        mean_ua = fps.gather_sum(cal.sum(u))/total_elements
        min_va = fps.gather_min(v.min())
        max_va = fps.gather_max(v.max())
        mean_va = fps.gather_sum(cal.sum(v))/total_elements
        min_wa = fps.gather_min(w.min())
        max_wa = fps.gather_max(w.max())
        mean_wa = fps.gather_sum(cal.sum(w))/total_elements

        #--- 检测密度气压温度
        min_rho = fps.gather_min(rho.min())
        max_rho = fps.gather_max(rho.max())
        mean_rho = fps.gather_sum(cal.sum(rho))/total_elements
        min_t = fps.gather_min(t.min())
        max_t = fps.gather_max(t.max())
        mean_t = fps.gather_sum(cal.sum(t))/total_elements
        min_p = fps.gather_min(p.min())
        max_p = fps.gather_max(p.max())
        mean_p = fps.gather_sum(cal.sum(p))/total_elements

        fps.main_print("                    Min            Max           Mean")
        fps.main_print("    rho:", "    %10.4f"%(min_rho), "    %10.4f"%(max_rho), "    %10.4f"%(mean_rho))  
        fps.main_print("    u  :", "    %10.4f"%(min_ua), "    %10.4f"%(max_ua), "    %10.4f"%(mean_ua))  
        fps.main_print("    v  :", "    %10.4f"%(min_va), "    %10.4f"%(max_va), "    %10.4f"%(mean_va))  
        fps.main_print("    w  :", "    %10.4f"%(min_wa), "    %10.4f"%(max_wa), "    %10.4f"%(mean_wa))  
        fps.main_print("    P  :", "    %10.4f"%(min_p), "    %10.4f"%(max_p), "    %10.4f"%(mean_p))  
        fps.main_print("    T  :", "    %10.4f"%(min_t), "    %10.4f"%(max_t), "    %10.4f"%(mean_t))  

        #--- 检测异常值
        if min_rho <=0 :
            fps.raise_error(f"density cannot be less than 0 kg/m^3!")
        if max_ua > 400 or min_ua < -400:
            fps.raise_error(f"ua cannot exceed the speed of sound!")
        if max_va > 400 or min_va < -400:
            fps.raise_error(f"va cannot exceed the speed of sound!")
        if max_wa > 400 or min_wa < -400:
            fps.raise_error(f"wa cannot exceed the speed of sound!")
        if min_t <0 :
            fps.raise_error(f"temperature cannot be less than 0 K!")
        if min_p <0 :
            fps.raise_error(f"pressure cannot be less than 0 Pa!")

        return