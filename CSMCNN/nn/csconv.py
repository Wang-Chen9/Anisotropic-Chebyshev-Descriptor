# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:31:30 2021

@author: Michael
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class ChebConv(nn.Module):
    def __init__(self,in_size,out_size,K=6,bias=True):
        super().__init__()
        self.in_size=in_size
        self.out_size=out_size
        self.K=K
        self.weight=Parameter(torch.Tensor(K,in_size,out_size))
    
        nn.init.xavier_uniform_(self.weight)
        
        if bias:
            self.bias=Parameter(torch.Tensor(out_size))
            nn.init.constant_(self.bias,0)
        else:
            self.register_parameter('bias',None)
        
    def forward(self,x,V,D,A):
        # x: input [N,P]
        # V: eigen-vectors [N,K]
        # D: eigen-values [K,1] in [-1,1] 计算采用乘法的广播
        # A: area [N,1] 计算采用乘法的广播
        D=2*D/D[-1]-1
        
        c=torch.matmul(V.t(),A*x) 
        
        Tx_0=c
        out=torch.matmul(Tx_0,self.weight[0])
        
        if self.K > 1:
            Tx_1=D*c 
            out=out+torch.matmul(Tx_1,self.weight[1])
            
        for k in range(2,self.K):
            Tx_2=2*D*Tx_1-Tx_0
            out=out+torch.matmul(Tx_2,self.weight[k])
            Tx_0,Tx_1=Tx_1,Tx_2
        
        if self.bias is not None:
            out=out+self.bias
        
        
        return torch.matmul(V,out)
