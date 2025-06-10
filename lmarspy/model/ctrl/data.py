#!/usr/bin/env python

# Purpose: Ctrl
# Author: Chen Research Group (xichen.me@outlook.com)

from lmarspy.model.atm import Atm
from lmarspy.model.ic import IC
from lmarspy.model.flag import Flag
from lmarspy.fps.data import FpsData

class CtrlData:

    def __init__(self, fps:FpsData, input=None, output=None):

        self.fps = fps

        import time
        import sys
        
        start_time = time.time()
        fps.main_print('#--- Initialization...')
        
        fps.main_print(f"# model input yaml file is {input}")
        
        fps.main_print("# read yaml to get parameters")
        #--- read yaml to get parameters
        fg = Flag(fps, file_name=input)
        self.fg = fg

        #--- assign if do_output
        self.do_output = fg.do_output

        self.do_diag = fg.do_diag

        fps.npz = fg.npz
    
        fps.main_print("# decompose domain")
        #--- decompose domain
        fps.decompose_domain(fg.global_npx-1, fg.global_npy-1, fg.ng, fg.px, fg.py)
        
        fps.main_print("# init Atm")
        #--- init Atm
        atm = Atm(fps, fg)
        atm.nc_file = output + ".nc"
        fps.main_print(f"    netcdf file outputs to {atm.nc_file}")
        self.atm = atm

        #--- pass time vars
        self.num_atmos_calls = atm.num_atmos_calls
        fps.num_atmos_calls = atm.num_atmos_calls

        #--- init IC
        ic = IC(fps, atm.mp)

        ic.assign_atm(fps, atm)

        #--- extend vars
        atm.ext_scalars()

        #--- output IC
        if self.do_output:

            if not fg.nc_parallel:

                if fps.rank == fps.main_rank:

                    atm.create_file_global()

                atm.create_file()

                atm.write_data(0)

            else:

                atm.create_file_global_parallel()

                atm.write_data_parallel(0)

        #--- 诊断
        atm.check_var0()

        end_time = time.time()
        execution_time = end_time - start_time
        fps.main_print("    &初始化时间：", "%.3f"%execution_time, "秒")
        fps.main_print()
        fps.main_print('#--- Start the main loop...')
        # 制作进度条
        step_times = []
        bar_length = 50
        self.bar_length = bar_length
        self.step_times = step_times
        if fps.rank == fps.main_rank:
            finsh = "▓" * 0 
            need_do = "-" * bar_length
            progress = 0
            print("{:>6d}/{}[{}->{}]{:>6.2f}%".format(0, self.num_atmos_calls, finsh, need_do, progress))
            print()
            sys.stdout.flush()

        return



