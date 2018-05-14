import random
import numpy as np
import cmath
import scipy
from scipy.integrate import quad, nquad
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
import os
os.chdir('/Users/tblantano/Desktop')

random.seed()

start_time = time.time()

#constants
hbar=6.62606957e-34 #J*s
m=9.10938356e-31 #kg
D=0.5 #m
l=0.5 #m
a=0.5e-3 #m
h=1e-3 #m
j=complex(0,1)
e=0
#list of inputs for the screen's coords
yb=np.arange(-30,30,0.01)

def complex_quad(func, a, b, **kwargs):
    def real_func(x):
        return scipy.real(func(x))
    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

def K(x1,x2):
    f=(x2-x1)**2
    return f

def yinflim_case1(x,bcoord):
    yc1=((D+l-x)*(a-bcoord))/l + bcoord
    return yc1

def ysuplim_case1(x,bcoord):
    yc2=((D+l-x)*(a+h-bcoord))/l + bcoord
    return yc2

def yinflim_case3(x):
    yc1=a*x/D
    return yc1

def ysuplim_case3(x):
    yc2=(a+h)*x/D
    return yc2

def int_Ybeforescreen(x1,bcoord,t,S,yinflim,ysuplim): #S for number of steps + 1
    N=(2*np.pi*j*hbar*t)/(m*(1+j*e))
    if S==2:
        I1=complex_quad(lambda y: (N**(-S))*cmath.exp((j*m*(1+j*e)/(2*hbar*t))*(K(x1,0)+K(l+D,x1)+K(y,0)+K(bcoord,y))),yinflim,ysuplim)
        return I1[0]
#--------------------------------- # ---------------- # ----------------------------------
        
#1-Step propagator
        

def int_Xcase1(bcoord):
    I1=complex_quad(lambda x: int_Ybeforescreen(x,bcoord,2,2,yinflim_case1(x,bcoord),ysuplim_case1(x,bcoord)),D-0.4,D)
    return I1[0]
    
def int_Xcase2(bcoord):
    I1=complex_quad(lambda x: int_Ybeforescreen(x,bcoord,2,2,-ysuplim_case1(x,-bcoord),-yinflim_case1(x,-bcoord)),0,D)
    return I1[0]
    

def int_Xcase3(bcoord):
    I1=complex_quad(lambda x: int_Ybeforescreen(x,bcoord,2,2,yinflim_case3(x),ysuplim_case3(x)),D,D+l)
    return I1[0]

def int_Xcase4(bcoord):
    I1=complex_quad(lambda x: int_Ybeforescreen(x,bcoord,2,2,-ysuplim_case3(x),-yinflim_case3(x)),D,D+l)
    return I1[0]

#Integrando solo en la rendija ----------- # ------------- # ------------ # --------------

def int_Xcase1R(bcoord):
    I1=int_Ybeforescreen(D,bcoord,2,2,a,a+h)
    return I1
    
def int_Xcase2R(bcoord):
    I1=int_Ybeforescreen(D,bcoord,2,2,-a-h,-a)
    return I1
    

def int_Xcase3R(bcoord):
    I1=int_Ybeforescreen(D,bcoord,2,2,a,a+h)
    return I1

def int_Xcase4R(bcoord):
    I1=int_Ybeforescreen(D,bcoord,2,2,-a-h,-a)
    return I1

# ----------------------- # --------------------- #------------------- # ---------------

def real_func(y,x,bcoord,S,t):
    N=(2*np.pi*j*hbar*2)/(m*(1+j*e))
    return scipy.real((N**(-S))*cmath.exp((j*m*(1+j*e)/(2*hbar*t))*(K(x,0)+K(l+D,x)+K(y,0)+K(bcoord,y))))

def imag_func(y,x,bcoord,S,t):
    N=(2*np.pi*j*hbar*2)/(m*(1+j*e))
    return scipy.imag((N**(-S))*cmath.exp((j*m*(1+j*e)/(2*hbar*t))*(K(x,0)+K(l+D,x)+K(y,0)+K(bcoord,y))))

def limy2_case1(x,bcoord,S,t):
    return [yinflim_case1(x,bcoord),ysuplim_case1(x,bcoord)]

def case1(bcoord): #S for number of steps + 1
    real_integral = nquad(real_func, [limy2_case1,[0, D]],args=(bcoord,2,2))
    imag_integral = nquad(imag_func, [limy2_case1,[0, D]],args=(bcoord,2,2))
    return real_integral[0] + 1j*imag_integral[0]


k=Pool(5)
#kernel=np.array(k.map(int_Xcase1,yb))
kernel1=np.array(k.map(int_Xcase1R,yb))+np.array(k.map(int_Xcase2R,yb))
kernel2=np.array(k.map(int_Xcase1R,yb))#+np.array(k.map(int_Xcase2,yb))+np.array(k.map(int_Xcase3R,yb))+np.array(k.map(int_Xcase4R,yb))
pbb1=(abs(kernel1)**2)
pbb2=(abs(kernel2)**2)
k.close()
#k.join()

#print(int_Ybeforescreen(D,1e-3,1e-3,2,2,yinflim_case1(D,1e-3),ysuplim_case1(D,1e-3)))
#print(int_Ybeforescreen(D/3,1e-3,-1e-3,2,2,-ysuplim_case1(D/3,-1e-3),-yinflim_case1(D/3,-1e-3)))
#print(prob_screen(2,2,yb))
plt.title("Double slit and Single slit")
plt.xlabel("Screen Coords(meters)")
plt.ylabel("Probability")
plt.plot(yb,pbb1/max(pbb1),'o')
plt.plot(yb,pbb2/max(pbb2),'o')

#index=list(pbb).index(max(pbb))
#plt.axvline(x=yb[index],color='purple')
plt.savefig("double&single1step.png")
print("--- %s seconds ---" % (time.time() - start_time))

#-------------------------- # --------------------- # --------------------------------------


