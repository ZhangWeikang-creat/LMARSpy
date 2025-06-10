#!/usr/bin/env python

# Purpose: Ctrl
# Author: Chen Research Group (xichen.me@outlook.com)

import time
import sys

from .data import CtrlData
from lmarspy.model.dyn import Dyn

class CtrlStep:

    @staticmethod
    def step(ctrl: CtrlData, na: int) -> None:

        fps = ctrl.fps
        #--- time update
        import datetime
        dt = datetime.timedelta(seconds=ctrl.atm.dt_atmos)
        ctrl.atm.time_atmos += dt

        bar_length = ctrl.bar_length
        step_times = ctrl.step_times
        fps.main_print()
        fps.main_print(f'*** stepping {na} :', ctrl.atm.time_atmos.strftime('%Y-%m-%d %H:%M:%S'))
        fps.comm.Barrier()
        start_time = time.time()
        
        #--- create dyn and perform step
        time0 = time.time()
        dyn = Dyn(ctrl.atm)
        dyn.step()
        fps.comm.Barrier()
        time1 = time.time()
        execution_time_dyn = time1 - time0
        fps.main_print(f"# 动力过程所需时间", "%.4f"%execution_time_dyn, "秒")
        fps.dyn_time.append(execution_time_dyn)

        time0 = time.time()
        #--- diag
        if ctrl.do_diag:
            ctrl.atm.check_var()
            #--- system info
            fps.main_print(f"# 内存使用统计")
            fps.get_memory()
        fps.comm.Barrier()
        time1 = time.time()
        execution_time_diag = time1 - time0
        fps.main_print(f"# 诊断过程所需时间", "%.4f"%execution_time_diag, "秒")
        fps.diag_time.append(execution_time_diag) 
        
        time0 = time.time()
        out_fre = ctrl.fg.out_fre
        tid = na//out_fre
        #--- output
        if ctrl.do_output:
            #--- 写入数据
            if tid == na/out_fre:
                if not ctrl.fg.nc_parallel:
                    ctrl.atm.write_data(tid)
                else:
                    ctrl.atm.write_data_parallel(tid)
            #--- 合并文件
            if na == ctrl.num_atmos_calls and not ctrl.fg.nc_parallel:
                fps.main_print(f"#--- 合并文件...")
                start_time = time.time()
                if fps.rank == fps.main_rank:
                    ctrl.atm.merge_ncfile()
                end_time = time.time()
                execution_time = end_time - start_time
                fps.main_print("    &合并文件时间：", "%.3f"%execution_time, "秒")
                fps.main_print()

        fps.comm.Barrier()
        time1 = time.time()
        execution_time_io = time1 - time0
        fps.main_print(f"# IO过程所需时间", "%.4f"%execution_time_io, "秒")
        fps.io_time.append(execution_time_io) 

        fps.comm.Barrier()
        end_time = time.time()
        # 计算执行时间
        step_times.append(end_time - start_time)
        execution_time = sum(step_times)
        progress = na/ctrl.num_atmos_calls*100
        step_time = execution_time/na
        nat = 10
        if na > nat:
            need_time = sum(step_times[-nat:])/nat*(ctrl.num_atmos_calls-na)/60
        else:
            need_time = step_time*(ctrl.num_atmos_calls-na)/60
        fps.main_print("# 进度统计：", "%.2f"%(progress), "%")
        fps.main_print("    平均单步时间：", "%.3f"%(step_time), "秒", "已运行总时间：", "%.3f"%(execution_time/60), "分钟")
        fps.main_print("    预计剩余时间：", "%.3f"%(need_time), "分钟", "预计总时间：", "%.3f"%(need_time+execution_time/60), "分钟")
        
        # 制作进度条
        if fps.rank == fps.main_rank:
            print(f'\033[{2}A', "\r", end="")
            finsh = "▓" * int(na/ctrl.num_atmos_calls*bar_length)
            need_do = "-" * (bar_length - int(na/ctrl.num_atmos_calls*bar_length))
            print("{:>6d}/{}[{}->{}]{:>6.2f}%".format(na, ctrl.num_atmos_calls, finsh, need_do, progress))
            print("平均单步：", "%.3f"%(step_time), "秒", "已运行：", "%.3f"%(execution_time/60), "分钟",\
                  "预计剩余：", "%.3f"%(need_time), "分钟", "预计总：", "%.3f"%(need_time+execution_time/60), "分钟", ""*10)
                
        sys.stdout.flush()

        return


