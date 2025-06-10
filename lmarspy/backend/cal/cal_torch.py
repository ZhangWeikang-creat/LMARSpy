#!/usr/bin/env python

# Purpose: General Scientific Computing Library cal: torch
# Author: Chen Research Group (xichen.me@outlook.com)

import torch
class cal:

    @staticmethod
    @torch.jit.script
    def arange(data):

        return torch.arange(data)  

    @staticmethod
    @torch.jit.script
    def pow(var, exp):

        return torch.exp(exp * torch.log(var))

    @staticmethod
    @torch.jit.script
    def unsqueeze(data, dim: int=0):

        return torch.unsqueeze(data, dim=dim)

    @staticmethod
    @torch.jit.script
    def exp(data):

        return torch.exp(data)         

    @staticmethod
    @torch.jit.script
    def log(data):

        return torch.log(data)


    @staticmethod
    @torch.jit.script
    def sqrt(data):
        
        return torch.sqrt(data)
    
    @staticmethod
    @torch.jit.script
    def abs(data):
        
        return torch.abs(data)
    
    @staticmethod
    @torch.jit.script
    def mean(data):
        
        return torch.mean(data)
    
    @staticmethod
    @torch.jit.script
    def sum(data):
        
        return torch.sum(data)
    
    @staticmethod
    @torch.jit.script
    def min(data1, data2):
        
        return torch.minimum(data1, data2)
    
    @staticmethod
    @torch.jit.script
    def max(data1, data2):
        
        return torch.maximum(data1, data2) 
    
    @staticmethod
    @torch.jit.script
    def sign(data1, data2):
        
        return ((data2>=0).int() - (data2<0).int()) * torch.abs(data1)

    @staticmethod
    @torch.jit.script
    def ifb(rlt):

        return rlt.int()

    @staticmethod
    @torch.jit.script
    def flip(data, dim: int=0):
        
        return torch.flip(data, dims=[dim])
    
    @staticmethod
    def cat(data, dim: int=0):
        
        return torch.cat(data, dim=dim)

    @staticmethod
    @torch.jit.script
    def cumsum(data, dim: int=0):
        
        return torch.cumsum(data, dim=dim)

    @staticmethod
    @torch.jit.script
    def diff(data, dim: int=0):
        
        return torch.diff(data, dim=dim) 

    @staticmethod
    @torch.jit.script
    def contiguous(data):
        
        return data.contiguous()
    
    @staticmethod
    @torch.jit.script
    def isnan(data):
        
        return torch.isnan(data)

    #--- 线性代数
    @staticmethod
    @torch.jit.script
    def linalg_solve(A, b):
        
        return torch.linalg.solve(A, b)
    
    @staticmethod
    @torch.jit.script
    def linalg_det(A):
        
        return torch.det(A)
    
    @staticmethod
    @torch.jit.script
    def linalg_inv(A):
        
        return torch.inverse(A)
    
    @staticmethod
    @torch.jit.script
    def matmul(A, B):
        
        return torch.matmul(A, B)
        