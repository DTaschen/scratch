'''
Created on 18.05.2012

@author: demian
'''
import numpy as np

import mayavi
from mayavi import mlab

[r,s] = np.mgrid[0.01:1:0.01,0.01:1:0.01]
d = (1-r)*np.log2((1-r)/(1-s)) + r*np.log2(r/s)
x = r
y = s
z = d
surface = mlab.mesh(x, y ,z)
