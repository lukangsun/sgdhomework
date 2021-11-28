# -*- coding: utf-8 -*-
"""
Created on Mon Sep  20 19:29:31 2021

@author: Lukang Sun
"""

from matplotlib.legend import Legend
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from numpy import matrix
from numpy import *
import math
import heapq
from random import sample
from numpy import linalg as la



def rank2(x):
    tmp = matrix([[x[0,0],x[1,0]],[x[2,0],x[3,0]]])
    dd=la.svd(tmp)
    tp = diag(dd[1])
    if abs(tp[0,0])>=abs(tp[1,1]):
        tp[1,1] = 0
    else:
        tp[0,0] = 0
    tm = dd[0]*tp*dd[2]
    return matrix([[tm[0,0]],[tm[0,1]],[tm[1,0]],[tm[1,1]]])

aa=matrix([[1],[2],[-4],[3]])

print(rank2(aa))

