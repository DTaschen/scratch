'''
Created on 16.05.2012

@author: demian
'''

import numpy as np
import sympy as sp

from scipy.linalg import \
     inv, det
from numpy import \
     array, zeros, int_, float_, ix_, dot, linspace, hstack, vstack, arange, \
     identity
     

def index_mapping_one_Node(dNx_mtx):    
     B_idx_map_one = ((0,1,2,2),(0,1,0,1))
     
     dN_idx_map_one =((0,1,1,0),(0,0,0,0))
     
     B = zeros ((3,2),dtype ='f')
         
     B[B_idx_map_one] = dNx_mtx[dN_idx_map_one]
     
     return B

''''Beispiel1'''

print "-----Beispiel one Node-------"

dNx_mtx = np.array([[1],[2]], dtype='f')

print "dNx_mtx = ", dNx_mtx

print "B-Matrix = ", index_mapping_one_Node(dNx_mtx)

print "-----Ende-----"

'''Ende Beispiel'''


def index_mapping_four_Node (dNx_mtx):
    
    B_idx_map_four = ((0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2),(0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7))
     
    dN_idx_map_four =((0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0),(0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3))
     
    B = zeros ((3,8),dtype ='f')
          
    B = zeros ((3,8),dtype ='f')

    B[B_idx_map_four] = dNx_mtx[dN_idx_map_four]
    
    return B


'''Beispiel 2'''

print "-----Beispiel four Node-------"

dNx_mtx4 = np.array([[1,1,1,1],[2,2,2,2]], dtype='f')

print "dNx_mtx = ", dNx_mtx4

print "B-Matrix = ", index_mapping_four_Node(dNx_mtx4)

print "-----Ende-----"

'''Ende Beispiel 2'''