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
from random import sample

#define list of a b and calculate some key parameter

a = [matrix([[0.94884523, 0.31257516]]), matrix([[0.64695759, 0.79089169]]), matrix([[0.70218109, 0.91473775]]), matrix([[0.03035042, 0.21034799]]), matrix([[0.99278455, 0.2554682 ]]), matrix([[0.16064759, 0.09062056]]), matrix([[0.41438167, 0.77718962]]), matrix([[0.4953842 , 0.93027311]]), matrix([[0.7692516 , 0.19772597]]), matrix([[0.12430258, 0.03779965]])]
b = [matrix([[0.4231786]]), matrix([[0.524466]]), matrix([[0.17981303]]), matrix([[0.50033184]]), matrix([[0.71116473]]), matrix([[0.0604399]]), matrix([[0.37100656]]), matrix([[0.91263449]]), matrix([[0.66417151]]), matrix([[0.65995097]])]
a_T=[]
for i in a:
    a_T.append(i.T)
a_Ttimesa=matrix([[0,0],[0,0]])
a_Ttimesb=matrix([[0],[0]])
for i in range(10):
    a_Ttimesa = a_Ttimesa + (a_T[i])*a[i]
    a_Ttimesb = a_Ttimesb + (b[i])[0,0]*(a_T[i])

x_min = -(a_Ttimesa.I)*a_Ttimesb
print(x_min)


sigma = 3993.739347850591
tau_best = 10
tau = 10
prob = 0.2
                                       
 #define each function  and its gradient

class func:
    def __init__(self, a,b):
        self.a = a
        self.b = b
        
    def value(self,x):
        return  0.5*((self.a*x+self.b)[0,0])**2+0.1*(math.sin(x[0,0]+0.07)+math.cos(x[1,0]+0.45))
                
    def grad(self,x):
        return ((self.a.T)*(self.a)*x+(self.a.T)*(self.b[0,0])+matrix([[0.1*math.cos(x[0,0]+0.07)],[-0.1*math.cos(x[1,0]+0.45)]]))


funct = []
for i in range(10):
    funct.append(func(a[i],b[i]))              

def fun_sum(x):
    sum = 0
    for j in range(10):
        sum +=funct[j].value(x)

    return 0.1*sum    
#nice tau sampling
def nicesam(tau):
    list1 = [0,1,2,3,4,5,6,7,8,9]                              
    return sample(list1,tau)

def bernouli(prob):
    tmp = random.uniform(0,1)
    if tmp<prob:
        return 1
    else:
        return 0    



#hyper parameter
stepsize = 1000**(-1)

init = matrix([[-0.5],[-0.2]])
x = 0
num_r = 10
dis = []
tauu = 3

#iteration



#Sparsification
def expect1(xx,yy,zz,dd,kk,jj,pp,pr):  
    tau = xx 
    iter = yy
    num_round = zz
    initial = dd
    min = kk
    stepsizes = jj
    dis = 0
    taau = pp
    probb = pr
    for rd in range(num_round):
        x = initial
        traj = []
        traj.append(x)
        for i in range(iter):
            index = nicesam(taau)
            gradi = matrix([[0],[0]])
            for ff in index:
                gradi =  gradi +  funct[ff].grad(x)
               
            x = x - stepsizes*(1/taau)*gradi   
            traj.append(x)
        dis = dis + (((x-min).T)*((x-min)))[0,0]
    return dis/num_round
#Sparsification
def expect3(xx,yy,zz,dd,kk,jj,pp,pr):  
    tau = xx 
    iter = yy
    num_round = zz
    initial = dd
    min = kk
    stepsizes = jj
    dis = 0
    taau = pp
    probb = pr

    for rd in range(num_round):
        x = initial
        y = x
        traj = []
        traj.append(x)
        for i in range(iter):
            if bernouli(probb):
                y = x
            index = nicesam(taau)
            gradi = matrix([[0],[0]])
            for ff in index:
                gradi =  gradi +  funct[ff].grad(x)-funct[ff].grad(y)
            gradi = (1/taau)*gradi
            for kk in range(10):
                gradi = gradi+0.1*funct[kk].grad(y)
            x = x - stepsizes*gradi   
            traj.append(x)
        dis = dis + (((x-min).T)*((x-min)))[0,0]
    return dis/num_round





dis_final1=[]
dis_final2=[]

iter = [500,1000,1500,2000,2500,3000]

for i in iter:
    print(i)
    #dis_final.append(expect(10,i,num_r,init,x_min,stepsize,0))  #SGD
    print(i)
    dis_final1.append(expect1(10,i,num_r,init,x_min,stepsize,5,0.2))#DCGD
    #print(i)
    #dis_final2.append(expect3(10,i,num_r,init,x_min,stepsize,5,0.2))#DIANA






#plot

plt.plot(iter,dis_final1,linewidth = 1,marker= '+',markersize = 10, label = 'DCGD')
#plt.plot(iter,dis_final2,linewidth = 1,marker= 'x',markersize = 10,label = 'DIANA')

 
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Error ')
#plt.title('tau = 5, p = 0.2')
plt.show()