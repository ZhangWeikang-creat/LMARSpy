from mpi4py import MPI
import numpy as np
from lmarspy.backend.field import Field
from lmarspy.backend.cal import cal
from .data import FpsData
from .base import FpsBase

class FpsBoundary:
    
    def exchange_boundary(fps: FpsData, local_data):
        
        rank = fps.rank
        comm = fps.comm

        x_start = fps.x_start
        x_end = fps.x_end
        y_start = fps.y_start
        y_end = fps.y_end

        px = fps.px
        py = fps.py
        nx = fps.nx
        ny = fps.ny

        npx = fps.npx
        npy = fps.npy

        ng = fps.ng
        global_npx = fps.global_npx
        global_npy = fps.global_npy
        npz = fps.npz
        
        dtype = np.float64
        
        if global_npy > 2:
            s_send = cal.contiguous(local_data[:, ng:2*ng, :])
            n_send = cal.contiguous(local_data[:, -2*ng:-ng:, :])

            s_recv = np.empty(s_send.shape, dtype=dtype)
            n_recv = np.empty(n_send.shape, dtype=dtype)
            

        if global_npx > 2:
            w_send = cal.contiguous(local_data[:, :, ng:2*ng])
            e_send = cal.contiguous(local_data[:, :, -2*ng:-ng])

            w_recv = np.empty(w_send.shape, dtype=dtype)
            e_recv = np.empty(e_send.shape, dtype=dtype)

        reqs = []
        if global_npy > 2:

            s_rank = rank - px if y_start > 0 else rank + px*(py-1)
            n_rank = rank + px if y_end < ny else rank - px*(py-1)
            n_tag = fps.get_next_tag()
            s_tag = fps.get_next_tag()

            reqs.append(comm.Isend(n_send, dest=n_rank, tag=n_tag))
            reqs.append(comm.Irecv(n_recv, source=n_rank, tag=s_tag))

            reqs.append(comm.Isend(s_send, dest=s_rank, tag=s_tag))
            reqs.append(comm.Irecv(s_recv, source=s_rank, tag=n_tag))

        if global_npx > 2:
            
            w_rank = rank - 1 if x_start > 0 else rank + px -1
            e_rank = rank + 1 if x_end < nx else rank - px + 1
            w_tag = fps.get_next_tag()
            e_tag = fps.get_next_tag()
        
            reqs.append(comm.Isend(w_send, dest=w_rank, tag=w_tag))
            reqs.append(comm.Irecv(w_recv, source=w_rank, tag=e_tag))

            reqs.append(comm.Isend(e_send, dest=e_rank, tag=e_tag))
            reqs.append(comm.Irecv(e_recv, source=e_rank, tag=w_tag))

        MPI.Request.Waitall(reqs)

        if global_npy > 2:
            local_data[:, -ng:, :] = Field(data=n_recv)
            local_data[:, :ng, :] = Field(data=s_recv)

        if global_npx > 2:
            local_data[:, :, :ng] = Field(data=w_recv)
            local_data[:, :, -ng:] = Field(data=e_recv) 

        return
    
    def exchange_boundary_create(fps: FpsData, send_data, w_recv, e_recv, s_recv, n_recv, reqs):
        
        rank = fps.rank
        subdomains = fps.subdomains
        comm = fps.comm

        x_start = fps.x_start
        x_end = fps.x_end
        y_start = fps.y_start
        y_end = fps.y_end

        px = fps.px
        py = fps.py
        nx = fps.nx
        ny = fps.ny

        npx = fps.npx
        npy = fps.npy

        ng = fps.ng
        global_npx = fps.global_npx
        global_npy = fps.global_npy
        npz = fps.npz
        
        #--- 指派收发任务
        if global_npy > 2:
            #--- 未来球立方网格要传输数据更新
            if py == 1:
                s_send = cal.contiguous(send_data[:, ng:2*ng, :])
                n_send = cal.contiguous(send_data[:, -2*ng:-ng:, :])

                send_data[:, :ng, :] = n_send
                send_data[:, -ng:, :] = s_send

            else:
                s_send = cal.contiguous(send_data[:, ng:2*ng, :])
                n_send = cal.contiguous(send_data[:, -2*ng:-ng:, :]) 

                s_rank = rank - px if y_start > 0 else rank + px*(py-1)
                n_rank = rank + px if y_end < ny else rank - px*(py-1)
                n_tag = fps.get_next_tag()
                s_tag = fps.get_next_tag()

                reqs.append(comm.Isend(n_send, dest=n_rank, tag=n_tag))
                reqs.append(comm.Irecv(n_recv, source=n_rank, tag=s_tag))

                reqs.append(comm.Isend(s_send, dest=s_rank, tag=s_tag))
                reqs.append(comm.Irecv(s_recv, source=s_rank, tag=n_tag))

                '''
                for i in range(fps.size): # 必须给出传输的顺序，不然通信会锁死， MPI没有这个问题，这样写会更慢
                    tag = fps.get_next_tag()
                    y_start = subdomains[i][2]
                    if y_start > 0:
                        s_rank = i - px
                    else:
                        s_rank = i + px*(py-1)
                    if fps.rank == i:
                        reqs.append(comm.Isend(s_send, dest=s_rank, tag=tag))
                    if fps.rank == s_rank:
                        reqs.append(comm.Irecv(n_recv, source=i, tag=tag))
                
                for i in range(fps.size):
                    tag = fps.get_next_tag()
                    y_end = subdomains[i][3]
                    if y_end < ny:
                        n_rank = i + px 
                    else:
                        n_rank = i - px*(py-1)
                    if fps.rank == i:
                        reqs.append(comm.Isend(n_send, dest=n_rank, tag=tag))
                    if fps.rank == n_rank:
                        reqs.append(comm.Irecv(s_recv, source=i, tag=tag))
                '''

        if global_npx > 2:
            #--- 未来球立方网格要传输数据更新
            if px == 1:
                w_send = cal.contiguous(send_data[:, :, ng:2*ng])
                e_send = cal.contiguous(send_data[:, :, -2*ng:-ng])

                send_data[:, :, :ng] = e_send
                send_data[:, :, -ng:] = w_send
            else:
                w_send = cal.contiguous(send_data[:, :, ng:2*ng])
                e_send = cal.contiguous(send_data[:, :, -2*ng:-ng])

                w_rank = rank - 1 if x_start > 0 else rank + px -1
                e_rank = rank + 1 if x_end < nx else rank - px + 1
                w_tag = fps.get_next_tag()
                e_tag = fps.get_next_tag()
            
                reqs.append(comm.Isend(w_send, dest=w_rank, tag=w_tag))
                reqs.append(comm.Irecv(w_recv, source=w_rank, tag=e_tag))

                reqs.append(comm.Isend(e_send, dest=e_rank, tag=e_tag))
                reqs.append(comm.Irecv(e_recv, source=e_rank, tag=w_tag))

                '''
                for i in range(fps.size):
                    tag = fps.get_next_tag()

                    x_start = subdomains[i][0]
                    if x_start > 0:
                        w_rank = i - 1
                    else:
                        w_rank = i +  px -1
                    if fps.rank == i:
                        reqs.append(comm.Isend(w_send, dest=w_rank, tag=tag))
                    if fps.rank == w_rank:
                        reqs.append(comm.Irecv(e_recv, source=i, tag=tag))
                
                for i in range(fps.size):
                    tag = fps.get_next_tag()
                    x_end = subdomains[i][1]
                    if x_end < nx:
                        e_rank = i + 1
                    else:
                        e_rank = i - px + 1
                    if fps.rank == i:
                        reqs.append(comm.Isend(e_send, dest=e_rank, tag=tag))
                    if fps.rank == e_rank:
                        reqs.append(comm.Irecv(w_recv, source=i, tag=tag))
                '''

        return
    
    def exchange_boundary_wait(fps: FpsData, send_data, w_recv, e_recv, s_recv, n_recv, reqs):

        if reqs == []:
            return

        px = fps.px
        py = fps.py

        ng = fps.ng
        global_npx = fps.global_npx
        global_npy = fps.global_npy

        #--- 等待收发完成
        for req in reqs:
            req.wait()

        #--- 边界值赋值
        if global_npy > 2 and py > 1:
            send_data[:, :ng, :] = s_recv
            send_data[:, -ng:, :] = n_recv

        if global_npx > 2 and px > 1:
            send_data[:, :, :ng] = w_recv
            send_data[:, :, -ng:] = e_recv

        reqs[:] = []

        return
    