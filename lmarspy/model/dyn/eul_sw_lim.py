
from lmarspy.model.const import Const
from lmarspy.model.sci import Sci

from .data import DynData
from lmarspy.backend.field import Field
if Field.backend == "numpy":
    from .util_numpy import DynUtil
elif Field.backend == "torch":
    from .util_torch import DynUtil
else:
    raise ValueError(f'Unsurpported backend: {Field.backend}')

class DynEulSwLim:

    @staticmethod
    def step(dyn: DynData, dt: float, ng: int = 1) -> None:

        fps = dyn.fps

        sw_dyn = dyn.sw_dyn
        #--- dim parameters
        npx = dyn.mp.npx
        npy = dyn.mp.npy
        npz = dyn.mp.npz

        ib  = dyn.mp.ib
        ie  = dyn.mp.ie
        ibd = dyn.mp.ibd
        ied = dyn.mp.ied

        jb  = dyn.mp.jb
        je  = dyn.mp.je
        jbd = dyn.mp.jbd
        jed = dyn.mp.jed

        #--- objs
        fg = dyn.fg
        mp = dyn.mp
        dg = dyn.dg

        #--- var pointers
        ua0  = dyn.ua0
        va0  = dyn.va0
        wa0  = dyn.wa0
        dp0  = dyn.dp0
        pt0  = dyn.pt0
        dz0  = dyn.dz0

        dp = dyn.dp
        pt = dyn.pt
        dz = dyn.dz
        ua = dyn.ua
        va = dyn.va
        wa = dyn.wa

        uaw = dyn.uaw
        uae = dyn.uae
        vas = dyn.vas 
        van = dyn.van 
        wato = dyn.wato
        wadn = dyn.wadn

        pr  = dyn.pr
        prw = dyn.prw
        pre = dyn.pre
        prs = dyn.prs
        prn = dyn.prn

        pa  = dyn.pa
        pato = dyn.pato
        padn = dyn.padn

        ut   = dyn.ut
        prcx = dyn.prcx
        ppcx = dyn.ppcx
        vt   = dyn.vt
        prcy = dyn.prcy
        ppcy = dyn.ppcy

        wt  = dyn.wt
        pae = dyn.pae
        ppe = dyn.ppe

        pe  = dyn.pe
        ph = dyn.ph

        wk = dyn.wk
        wke = dyn.wke

        fx = dyn.fx
        fy = dyn.fy
        fz = dyn.fz

        fxw = dyn.fxw
        fxe = dyn.fxe
        fys = dyn.fys
        fyn = dyn.fyn
        fzto = dyn.fzto
        fzdn = dyn.fzdn

        fxtr = dyn.fxtr
        fytr = dyn.fytr
        fztr = dyn.fztr

        fxtrw = dyn.fxtrw
        fxtre = dyn.fxtre
        fytrs = dyn.fytrs
        fytrn = dyn.fytrn
        fztrto = dyn.fztrto
        fztrdn = dyn.fztrdn

        rhs_ua = dyn.rhs_ua
        rhs_va = dyn.rhs_va
        rhs_wa = dyn.rhs_wa
        rhs_dp = dyn.rhs_dp
        rhs_pt = dyn.rhs_pt
        rhs_dz = dyn.rhs_dz

        crx = dyn.crx
        xfx = dyn.xfx
        cry = dyn.cry
        yfx = dyn.yfx
        crz = dyn.crz
        zfx = dyn.zfx

        #--- mpi
        dp_reqs = dyn.dp_reqs
        dz_reqs = dyn.dz_reqs
        pt_reqs = dyn.pt_reqs
        ua_reqs = dyn.ua_reqs
        va_reqs = dyn.va_reqs
        wa_reqs = dyn.wa_reqs

        dp_w_recv = dyn.dp_w_recv
        dz_w_recv = dyn.dz_w_recv
        pt_w_recv = dyn.pt_w_recv
        ua_w_recv = dyn.ua_w_recv
        va_w_recv = dyn.va_w_recv
        wa_w_recv = dyn.wa_w_recv

        dp_e_recv = dyn.dp_e_recv
        dz_e_recv = dyn.dz_e_recv
        pt_e_recv = dyn.pt_e_recv
        ua_e_recv = dyn.ua_e_recv
        va_e_recv = dyn.va_e_recv
        wa_e_recv = dyn.wa_e_recv

        dp_s_recv = dyn.dp_s_recv
        dz_s_recv = dyn.dz_s_recv
        pt_s_recv = dyn.pt_s_recv
        ua_s_recv = dyn.ua_s_recv
        va_s_recv = dyn.va_s_recv
        wa_s_recv = dyn.wa_s_recv

        dp_n_recv = dyn.dp_n_recv
        dz_n_recv = dyn.dz_n_recv
        pt_n_recv = dyn.pt_n_recv
        ua_n_recv = dyn.ua_n_recv
        va_n_recv = dyn.va_n_recv
        wa_n_recv = dyn.wa_n_recv

        lim_deg = dyn.lim_deg

        #--- start algorithm
        #--- calculate pa
        fps.exchange_boundary_wait(dp, dp_w_recv, dp_e_recv, dp_s_recv, dp_n_recv, dp_reqs)
        pr[:] = dp
        dz[:] = -dp/Const.GRAV

        if mp.npx > 2:
            DynUtil.get_var_we_lim(pr, prw, pre, mp, ng)
            fps.exchange_boundary_wait(ua, ua_w_recv, ua_e_recv, ua_s_recv, ua_n_recv, ua_reqs)
            #--- perform x-dir and y-dir interp with ng
            DynUtil.get_var_we(ua, uaw, uae, mp, ng)
            #--- get x-direction LMARS ut, prc
            DynUtil.get_lmars_x(ut, prcx, ppcx, uaw, uae, prw, pre, dyn.cs, dp, dz, wk, ib, ie, jb, je)
            #--- get crx
            DynUtil.get_xfx(xfx, ut, dt, dg.c_dy[jb:je,ib:ie+1])

        if mp.npy > 2:
            DynUtil.get_var_sn_lim(pr, prs, prn, mp, ng)
            fps.exchange_boundary_wait(va, va_w_recv, va_e_recv, va_s_recv, va_n_recv, va_reqs)
            #--- perform x-dir and y-dir interp with ng
            DynUtil.get_var_sn(va, vas, van, mp, ng)
            #--- get y-direction LMARS vt, prc
            DynUtil.get_lmars_y(vt, prcy, ppcy, vas, van, prs, prn, dyn.cs, dp, dz, wk, ib, ie, jb, je)
            #--- get cry
            DynUtil.get_yfx(yfx, vt, dt, dg.d_dx[jb:je+1,ib:ie])

        #=== advect dp
        rhs_dp[:] = 0
        if mp.npx > 2:
            #--- get fx
            DynUtil.get_var_we_lim(dp, fxw, fxe, mp, ng)
            DynUtil.get_x_upwind(fx, fxw, fxe, ut)
            fx[:] = fx*xfx
        if mp.npy > 2:
            #--- get fy
            DynUtil.get_var_sn_lim(dp, fys, fyn, mp, ng)
            DynUtil.get_y_upwind(fy, fys, fyn, vt)
            fy[:] = fy*yfx
        #--- update dp
        DynUtil.update_eul(rhs_dp, fx, fy, fz, dz[:,jb:je,ib:ie], dg.a_rda[jb:je,ib:ie])
        dp[:,jb:je,ib:ie] = dp0[:,jb:je,ib:ie] + rhs_dp
        fps.exchange_boundary_create(dp, dp_w_recv, dp_e_recv, dp_s_recv, dp_n_recv, dp_reqs)

        #=== advect ua(这个成功以后改成 vec inv 形式)
        if mp.npx > 2:
            rhs_ua[:] = 0
            #--- get fxtr
            DynUtil.get_var_we_lim(ua, fxtrw, fxtre, mp, ng)
            DynUtil.get_x_upwind(fxtr, fxtrw, fxtre, ut)
            if mp.npy > 2:
                #--- get fytr
                DynUtil.get_var_sn_lim(ua, fytrs, fytrn, mp, ng)
                DynUtil.get_y_upwind(fytr, fytrs, fytrn, vt)
            #--- update ua
            DynUtil.update_eul(rhs_ua, fxtr*fx, fytr*fy, fztr*fz, dz[:,jb:je,ib:ie], dg.a_rda[jb:je,ib:ie])
            #=== apply pgrad x
            DynUtil.apply_pgrad_x_eul(rhs_ua, prcx, dz[:,jb:je,ib:ie], dg.a_rdx[jb:je,ib:ie], dt)
            ua[:,jb:je,ib:ie] = (ua0[:,jb:je,ib:ie]*dp0[:,jb:je,ib:ie] + rhs_ua) / dp[:,jb:je,ib:ie]
            fps.exchange_boundary_create(ua, ua_w_recv, ua_e_recv, ua_s_recv, ua_n_recv, ua_reqs)

        #=== advect va(这个成功以后改成 vec inv 形式)
        if mp.npy > 2:
            rhs_va[:] = 0
            if mp.npx > 2:
                #--- get fxtr
                DynUtil.get_var_we_lim(va, fxtrw, fxtre, mp, ng)
                DynUtil.get_x_upwind(fxtr, fxtrw, fxtre, ut)
            #--- get fytr
            DynUtil.get_var_sn_lim(va, fytrs, fytrn, mp, ng)
            DynUtil.get_y_upwind(fytr, fytrs, fytrn, vt)
            #--- update va
            DynUtil.update_eul(rhs_va, fxtr*fx, fytr*fy, fztr*fz, dz[:,jb:je,ib:ie], dg.a_rda[jb:je,ib:ie])
            #=== apply pgrad y
            DynUtil.apply_pgrad_y_eul(rhs_va, prcy, dz[:,jb:je,ib:ie], dg.a_rdy[jb:je,ib:ie], dt)
            va[:,jb:je,ib:ie] = (va0[:,jb:je,ib:ie]*dp0[:,jb:je,ib:ie] + rhs_va) / dp[:,jb:je,ib:ie]
            fps.exchange_boundary_create(va, va_w_recv, va_e_recv, va_s_recv, va_n_recv, va_reqs) 

        dyn.count += 1
        if dyn.count == dyn.max_count:
            fps.exchange_boundary_wait(dp, dp_w_recv, dp_e_recv, dp_s_recv, dp_n_recv, dp_reqs)
            fps.exchange_boundary_wait(ua, ua_w_recv, ua_e_recv, ua_s_recv, ua_n_recv, ua_reqs)
            fps.exchange_boundary_wait(va, va_w_recv, va_e_recv, va_s_recv, va_n_recv, va_reqs)

        return        

