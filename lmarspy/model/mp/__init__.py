#!/usr/bin/env python

# Purpose: Communication and data decomposition tools
# Author: Chen Research Group (xichen.me@outlook.com)

class Mp:

    def __init__(self, npx: int, npy: int, npz: int, ng: int):

        #--- dimensions
        self.npx = npx
        self.npy = npy
        self.npz = npz
        self.ng  = ng

        if npx==2:
            ngx = 0
        else:
            ngx = ng
        self.ibd = 0
        self.ib  = self.ibd + ngx
        self.ie  = self.ib  + npx-1
        self.ied = self.ie  + ngx

        if npy==2:
            ngy = 0
        else:
            ngy = ng
        self.jbd = 0
        self.jb  = self.jbd + ngy
        self.je  = self.jb  + npy-1
        self.jed = self.je  + ngy

        self.dim_ha = [self.jed, self.ied]
        self.dim_va = [npz, self.jed, self.ied]

        self.dim_hb = [self.jed+1, self.ied+1]
        self.dim_vb = [npz, self.jed+1, self.ied+1]

        self.dim_hc = [self.jed, self.ied+1]
        self.dim_vc = [npz, self.jed, self.ied+1]

        self.dim_hd = [self.jed+1, self.ied]
        self.dim_vd = [npz, self.jed+1, self.ied]

        self.dim_vp = [npz+1, self.jed, self.ied]

        self.offset_x = 0
        self.offset_y = 0

        return


