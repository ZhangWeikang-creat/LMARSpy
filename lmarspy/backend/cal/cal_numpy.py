#!/usr/bin/env python

# Purpose: General Scientific Computing Library cal: numpy
# Author: Chen Research Group (xichen.me@outlook.com)

import numpy as np
from numba import jit

class cal:

    @staticmethod
    @jit(nopython=True)
    def arange(data):

        return np.arange(data)

    @staticmethod
    @jit(nopython=True)
    def pow(var, exp):

        return np.exp(exp * np.log(var))

    @staticmethod
    def unsqueeze(data, dim=0):

        return np.expand_dims(data, axis=dim)

    @staticmethod
    @jit(nopython=True)
    def exp(data):

        return np.exp(data)

    @staticmethod
    @jit(nopython=True)
    def log(data):

        return np.log(data)

    @staticmethod
    @jit(nopython=True)
    def sqrt(data):

        return np.sqrt(data)
    
    @staticmethod
    @jit(nopython=True)
    def abs(data):

        return np.abs(data)
    
    
    @staticmethod
    @jit(nopython=True)
    def mean(data):

        return np.mean(data)
    
    @staticmethod
    @jit(nopython=True)
    def sum(data):

        return np.sum(data)
    
    @staticmethod
    @jit(nopython=True)
    def min(data1, data2):

        return np.minimum(data1, data2)
    
    @staticmethod
    @jit(nopython=True)
    def max(data1, data2):

        return np.maximum(data1, data2)
    
    @staticmethod
    @jit(nopython=True)
    def sign(data1, data2):

        return ((data2>=0).astype(np.int32) - (data2<0).astype(np.int32)) * np.abs(data1)

    @staticmethod
    @jit(nopython=True)
    def ifb(rlt):

        return rlt.astype(np.int32)

    @staticmethod
    @jit(nopython=True)
    def flip(data, dim=0):

        return np.flip(data, axis=dim)
    
    @staticmethod
    @jit(nopython=True)
    def cat(data, dim=0):

        return np.concatenate(data, axis=dim)

    @staticmethod
    def cumsum(data, dim=0):

        return np.cumsum(data, axis=0)
    
    @staticmethod
    def diff(data, dim=0):

        return np.diff(data, axis=0)
    
    @staticmethod
    @jit(nopython=True)
    def contiguous(data):

        return np.ascontiguousarray(data)
    
    @staticmethod
    @jit(nopython=True)
    def isnan(data):

        return np.isnan(data)

    #--- 线性代数
    @staticmethod
    def linalg_solve(A, b):

        return np.linalg.solve(A, b)
    
    @staticmethod
    def linalg_det(A):

        return np.linalg.det(A)
    
    @staticmethod
    def linalg_inv(A):

        return np.linalg.inv(A)
    
    @staticmethod
    def matmul(A, B):

        return np.matmul(A, B)
    
    