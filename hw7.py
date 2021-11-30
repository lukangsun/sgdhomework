# -*- coding: utf-8 -*-
"""
Created on Mon Sep  20 19:29:31 2021

@author: Lukang Sun
"""
import matplotlib
from matplotlib.legend import Legend
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from numpy import matrix
from numpy import *
import math
import heapq
import random
from numpy import linalg as la
innitialpoint = matrix([[1],[1]])
a=5
b=3
c=2
d=7
#function
def f_1(x):
    return a*x[0,0]**2+b*x[1,0]**2+math.sin(x[0,0])

def f_2(x):
    return c*x[0,0]**2+d*x[1,0]**2+math.cos(x[1,0])

def grad_1(x):
    return (matrix([[2*a*x[0,0+math.cos(x[0,0])]],[2*b*x[1,0]]]))

def grad_2(x):
    return (matrix([[2*c*x[0,0]],[2*d*x[1,0]-math.sin(x[1,0])]]))

def hess_1(x):
    return matrix([[2*a-math.sin(x[0,0]),0],[0,2*b]])

def hess_2(x):
    return matrix([[2*c,0],[0,2*d-math.cos(x[1,0])]])

def unisam1():
    return random.choice([0,1])



initial = matrix([[1],[1]])
w1 = initial
w2 = initial

hessinv = la.inv(hess_1(innitialpoint))
print(grad_2(w2)-grad_1(w1))
