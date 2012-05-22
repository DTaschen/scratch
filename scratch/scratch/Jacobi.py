'''
Created on 14.05.2012

@author: demian
'''
import numpy as np
import sympy as sp
from scipy.linalg import \
     inv, det
from numpy import \
     array, zeros, int_, float_, ix_, dot, linspace, hstack, vstack, arange, \
     identity
     


x_,y_ = sp.symbols('x,y')

A= np.array([2*x_,y_**2,x_*y_**3])
print A
A_mtx= sp.Matrix(A)
print "____________________"
dA= A_mtx.jacobian([x_,y_])
print dA

