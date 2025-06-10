from colorama import Fore, Style
import psutil
import sys
import torch
import platform

from lmarspy.backend.field import Field
from lmarspy.backend.cal import cal
from .data import FpsData

class FpsBase:

    @staticmethod
    def print(fps: FpsData, args) -> None:

        for i in range(fps.size):
            if fps.rank == i:
                with open(fps.log_path, 'a', encoding='utf-8') as f:

                    f.write(f"From rank {fps.rank} of {fps.size}: ")
                    for arg in args:
                        f.write(f"{arg} ")
                    f.write("\n")
            
            fps.comm.Barrier()

        return
    
    @staticmethod
    def main_print(fps: FpsData, args) -> None:
        
        if fps.rank == fps.main_rank:
            with open(fps.log_path, 'a', encoding='utf-8') as f:

                for arg in args:
                    f.write(f"{arg} ")
                f.write("\n")
        
        fps.comm.Barrier()

        return
    
    @staticmethod
    def raise_error(fps: FpsData, info: str) -> None:
        
        print(f"from rank {fps.rank},", Fore.RED + "Fatal error:"+ Style.RESET_ALL, info)

        raise ValueError()
    
    @staticmethod
    def get_next_tag(fps):

        fps.current_tag += 1
        if fps.current_tag > fps.max_tag-100000:
            fps.current_tag = 0
        
        return fps.current_tag
    
    @staticmethod
    def get_memory(fps: FpsData):

        if Field.backend == "torch":
            if Field.device == "gpu":
                
                device = torch.device("cuda")
                allocated_memory = torch.cuda.max_memory_allocated(device)
                reserved_memory = torch.cuda.max_memory_reserved(device)
                total_memory = torch.cuda.get_device_properties(device).total_memory
                memory_percent = reserved_memory / total_memory *100

                FpsBase.print(fps, [f"{Field.get_device()}: 最大已分配 {allocated_memory / 1024 ** 3:.4f} GB", \
                                f"最大预留 {reserved_memory / 1024 ** 3:.4f} GB", \
                                f"总 {total_memory / 1024 ** 3:.4f} GB",\
                                f"内存使用率: {memory_percent:.2f}%"])
            else:
                memory_info = psutil.virtual_memory()
                total_memory = memory_info.total / (1024 ** 3)
                memory_max_info = fps.process.memory_info()
                max_use_memory = fps.gather_sum(Field(data=memory_max_info.rss / (1024 ** 3)))
                memory_percent = max_use_memory / total_memory *100
                cpu_brand = platform.processor()
                use_percent = fps.gather_sum(Field(data=psutil.cpu_percent(interval=0.1)))*100

                FpsBase.main_print(fps, [f"{cpu_brand}" + f"(CPU使用率:{use_percent:.2f}%):",\
                    f"最大已使用: {max_use_memory:.4f} GB", f"总: {total_memory:.4f} GB", f"内存使用率: {memory_percent:.2f}%"])
        
        if Field.backend == "numpy":

            memory_info = psutil.virtual_memory()
            total_memory = memory_info.total / (1024 ** 3)
            memory_max_info = fps.process.memory_info()
            max_use_memory = fps.gather_sum(Field(data=memory_max_info.rss / (1024 ** 3)))
            memory_percent = max_use_memory / total_memory *100
            cpu_brand = platform.processor()
            use_percent = fps.gather_sum(Field(data=psutil.cpu_percent(interval=0.1)))*100

            FpsBase.main_print(fps, [f"{cpu_brand}" + f"(CPU使用率:{use_percent:.2f}%):",\
                f"最大已使用: {max_use_memory:.4f} GB", f"总: {total_memory:.4f} GB", f"内存使用率: {memory_percent:.2f}%"])
    
    def init_model(fps: FpsData):

        fps.main_print()

        import datetime
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        fps.main_print("&开始时间&:", formatted_time)
        fps.main_print()

        fps.main_print("#--- backend configurations")
        fps.main_print(f"    backend: {Field.backend}")
        fps.main_print(f"    device: {Field.device}")

        fps.main_print("#--- used device")
        fps.main_print(f"    {Field.get_device()}")
        fps.main_print()

        fps.main_print("#--- FPC configurations")
        fps.main_print(f"    MASTER_ADDR set to: {fps.ip_address}")
        fps.main_print(f"    MASTER_PORT set to: {fps.free_port}")
        fps.main_print(f"    WORLD_SIZE set to: {fps.size}")
        fps.main_print()
        
        return
        
        
    def end_model(fps: FpsData):

        import datetime
        if Field.backend == 'torch':
            import torch.distributed as dist
            dist.destroy_process_group()

         #--- check dyn io time
        fps.main_print("#--- Overall statistics of dyn and io time...")
        fps.main_print(f"# dyn time")
        fps.main_print(f"min ", "%.3f"%(min(fps.dyn_time)), f"秒, max ", "%.3f"%(max(fps.dyn_time)), f"秒", f"mean ", \
                    "%.3f"%(sum(fps.dyn_time)/fps.num_atmos_calls), f"秒", f"合计:", "%.3f"%(sum(fps.dyn_time)/60), f"分钟")
        fps.main_print(f"# IO time")
        fps.main_print(f"min ", "%.3f"%(min(fps.io_time)), f"秒, max ", "%.3f"%(max(fps.io_time)), f"秒", f"mean ", \
                    "%.3f"%(sum(fps.io_time)//fps.num_atmos_calls), f"秒", f"合计:", "%.3f"%(sum(fps.io_time)/60), f"分钟")
        fps.main_print(f"# diag time")
        fps.main_print(f"min ", "%.3f"%(min(fps.diag_time)), f"秒, max ", "%.3f"%(max(fps.diag_time)), f"秒", f"mean ", \
                    "%.3f"%(sum(fps.diag_time)//fps.num_atmos_calls), f"秒", f"合计:", "%.3f"%(sum(fps.diag_time)/60), f"分钟")
        fps.main_print()
        
        fps.main_print('#---Completion...')
        fps.main_print()

        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        fps.main_print("&结束时间&:", formatted_time)
        fps.main_print()

                    