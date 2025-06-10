from mpi4py import MPI
import socket
import psutil
import os
from lmarspy.backend.field import Field

class FpsData:
    def __init__(self, output: str):
        
        #--- create log file
        self.log_path = output + ".log"
        with open(self.log_path, 'w') as f:
            f.write("**********日志文件**********\n")
                  
        #--- 初始化MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        main_rank = 0

        self.comm = comm
        self.rank = rank
        self.size = size
        self.main_rank = 0

        #--- 初始化分布式环境
        if rank == main_rank:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                free_port = s.getsockname()[1]
            master_info = (ip_address, free_port)
            comm.bcast(master_info, root=main_rank)
        else:
            master_info = comm.bcast(None, root=main_rank)

        ip_address, free_port = master_info
        self.ip_address = ip_address
        self.free_port = free_port

        #--- 设置必要的环境变量（使用 env:// 作为 init_method）
        os.environ['MASTER_ADDR'] = ip_address
        os.environ['MASTER_PORT'] = str(free_port)
        os.environ['WORLD_SIZE'] = str(size)
        os.environ['RANK'] = str(rank)

        local_rank = int(os.environ.get('SLURM_LOCALID', rank))
        self.local_rank = local_rank
        os.environ['LOCAL_RANK'] = str(local_rank)

        if Field.backend == "torch":
            #--- 初始化torch分布式环境
            import torch
            import torch.distributed as dist
            if Field.device == "gpu":
                dist.init_process_group(backend='nccl', init_method='env://')
                torch.cuda.set_device(local_rank)
            else:
                dist.init_process_group(backend='gloo', init_method='env://')
        #--- 初始化标签值
        self.current_tag = 0
        max_tag = comm.Get_attr(MPI.TAG_UB)
        self.max_tag = max_tag

        #--- to get values(未来转到mp)
        self.subdomains = None
        self.ng = None

        self.npx = None
        self.npy = None

        self.global_npx = None
        self.global_npy = None

        self.npz = None

        self.global_ibd = None
        self.global_ib = None
        self.global_ie = None
        self.global_ied = None

        self.x_start = None
        self.x_end = None
        self.y_start = None
        self.y_end = None

        self.px = None
        self.py = None
        self.nx = None
        self.ny = None

        self.dyn_time = []
        self.io_time = []
        self.diag_time = []

        #--- 进程编号
        pid = os.getpid()
        process = psutil.Process(pid)
        self.process = process
        
        
