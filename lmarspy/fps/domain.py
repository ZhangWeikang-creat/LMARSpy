from .data import FpsData
from .base import FpsBase

class FpsDomain:
    
    def decompose_domain(fps:FpsData, nx, ny, ng, px, py):
        """
        Decompose the domain into subdomains for each process.
        :param nx: Total number of grid points in x-direction
        :param ny: Total number of grid points in y-direction
        :param px: Number of processes in x-direction
        :param py: Number of processes in y-direction
        :return: List of (x_start, x_end, y_start, y_end) for each process
        """

        if px*py != fps.size:
            FpsBase.raise_error(fps, f'px * py != number of MPI processes')

        if nx < px:
            FpsBase.raise_error(fps, f'nx <= px, Number of processes in x-direction is less than Total number of grid points in x-direction')

        if ny < py:
            FpsBase.raise_error(fps, f'ny <= py, Number of processes in y-direction is less than Total number of grid points in y-direction')

        fps.px = px
        fps.py = py

        fps.nx = nx
        fps.ny = ny

        subdomains = []
        if nx == nx//px*px:
            pnx = nx//px
        else:
            pnx = nx//px + 1
        #--- 防止pnx过小时x_end超过nx
        if pnx < px:
            pnx -= 1

        if ny == ny//py*py:
            pny = ny//py
        else:
            pny = ny//py + 1
        if pny < py:
            pny -= 1

        for i in range(py):
            for j in range(px):

                x_start = j * pnx
                x_end = (j + 1) * pnx if j != px - 1 else nx
                y_start = i * pny
                y_end = (i + 1) * pny if i != py - 1 else ny
                subdomains.append((x_start, x_end, y_start, y_end))
        
        fps.subdomains = subdomains

        x_start, x_end, y_start, y_end = fps.subdomains[fps.rank]
        npx = x_end - x_start + 1
        npy = y_end - y_start + 1

        if npx == 2:
            ngx = 0
        else:
            ngx = ng

        if npy == 2:
            ngy = 0
        else:
            ngy = ng

        global_ibd = x_start
        global_ib = global_ibd + ngx
        global_ie = global_ib + npx -1
        global_ied = global_ie + ngx

        global_jbd = y_start
        global_jb = global_jbd + ngy
        global_je = global_jb + npy - 1
        global_jed = global_je + ngy

        fps.ng = ng
        fps.npx = npx
        fps.npy = npy
        fps.global_npx = nx+1
        fps.global_npy = ny+1

        fps.x_start = x_start
        fps.x_end = x_end
        fps.y_start = y_start
        fps.y_end = y_end

        fps.global_ibd = global_ibd
        fps.global_ib = global_ib
        fps.global_ie = global_ie
        fps.global_ied = global_ied

        fps.global_jbd = global_jbd
        fps.global_jb = global_jb
        fps.global_je = global_je
        fps.global_jed = global_jed

        for i in range(fps.size):
            fps.main_print(f"    processor {i}, x-direction: {subdomains[i][0]}->{subdomains[i][1]} y-direction: {subdomains[i][2]}->{subdomains[i][3]}")

        return subdomains
