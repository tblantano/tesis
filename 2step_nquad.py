import random
import numpy as np
import cmath
import scipy
from scipy.integrate import quad, dblquad, nquad
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
#import vegas

random.seed()

start_time = time.time()

#constants
hbar=1#6.62606957e-34 #J*s
m=1 #9.10938356e-34 #kg
D=0.5 #m
l=0.5 #m
a=0.5e-1 #m
h=1e-1 #m
e=1e-2
j=complex(0,1)

def myfunc(a):
    time.sleep(random.random())
    return a ** 2

def K(x1,x2):
    f=(x2-x1)**2
    return f

def yinflim_case1(x2,bcoord):
    yc1=((D+l-x2)*(a-bcoord))/l + bcoord
    return yc1

def ysuplim_case1(x2,bcoord):
    yc2=((D+l-x2)*(a+h-bcoord))/l + bcoord
    return yc2

def yinflim_case3(x2,y1,x1):
    yinf2=(a*(x2-x1)-y1*(x2-D))/(D-x1)
    return yinf2

def ysuplim_case3(x2,y1,x1):
    ysup2=((a+h)*(x2-x1)-y1*(x2-D))/(D-x1)
    return ysup2
    
def yinflim_case5(x):
    yc1=a*x/D
    return yc1

def ysuplim_case5(x):
    yc2=(a+h)*x/D
    return yc2

def x21(y1,x1,bcoord):
    return (-h*l*x1+a*((D**2)+D*(l-x1)-2*l*x1)+(D**2)*(h-bcoord)+D*(h*l-h*x1-l*y1+x1*bcoord))/(a*(D-l-x1)-h*x1-l*y1+D*(h-bcoord)+x1*bcoord)

def x22(y1,x1,bcoord):
    return (a*((D**2)+D*(l-x1)-2*l*x1)-D*(l*y1+bcoord*(D-x1)))/(a*(D-l-x1)-l*y1+bcoord*(x1-D))

def x23(y1,x1,bcoord):
    return (2*h*l*x1-a*((D**2)+D*(l-x1)-2*l*x1)-(D**2)*(h-bcoord)+D*(-h*l+h*x1+l*y1-bcoord*x1))/(h*l-a*(D-l-x1)+h*x1-D*(h-bcoord)+l*y1-bcoord*x1)

def x24(y1,x1,bcoord):
    return (h*l*x1-a*((D**2)+D*(l-x1)-2*l*x1)+D*(l*y1+bcoord*(D-x1)))/(h*l-a*(D-l-x1)+l*y1+bcoord*(D-x1))

def x21_13(y1,x1,bcoord):
    return (h*l*x1-a*((D**2)+D*(l-x1)-2*l*x1)-D*(l*y1+D*bcoord-x1*bcoord))/(h*l-a*(D-l-x1)-l*y1-D*bcoord+x1*bcoord)

def x22_13(y1,x1,bcoord):
    return (a*((D**2)+D*(l-x1)-2*l*x1)+D*(l*y1+bcoord*(D-x1)))/(a*(D-l-x1)+l*y1+bcoord*(D-x1))

def x23_13(y1,x1,bcoord):
    return (2*h*l*x1-a*((D**2)+D*(l-x1)-2*l*x1)-(D**2)*(h+bcoord)+D*(-h*l+h*x1-l*y1+bcoord*x1))/(h*l-a*(D-l-x1)+h*x1-D*(h+bcoord)-l*y1+bcoord*x1)

def x24_13(y1,x1,bcoord):
    return (-h*l*x1+a*((D**2)+D*(l-x1)-2*l*x1)+(D**2)*(h+bcoord)+D*(h*l-h*x1+l*y1-x1*bcoord))/(a*(D-l-x1)-h*x1+l*y1-x1*bcoord+D*(h+bcoord))

def x23_a(y1,x1,bcoord):
    return (a*D*(D+l-x1)-h*l*x1+(D**2)*(h+bcoord)+D*(h*l-x1*h+l*y1-x1*bcoord))/(a*(D+l-x1)-h*x1+l*y1-x1*bcoord+D*(h+bcoord))

def x23_b(y1,x1,bcoord):
    return (a*D*(D+l-x1)+h*l*x1+D*(l*y1+D*bcoord-x1*bcoord))/(h*l+a*(D+l-x1)+l*y1+D*bcoord-x1*bcoord)

def x23_d(y1,x1,bcoord):
    return (a*D*(D+l-x1)+h*l*x1-D*(l*y1+D*bcoord-x1*bcoord))/(h*l+a*(D+l-x1)-l*y1-D*bcoord+x1*bcoord)

def x23_e(y1,x1,bcoord):
    return (a*D*(D+l-x1)-h*l*x1+(D**2)*(h-bcoord)+D*(h*l-h*x1-l*y1+x1*bcoord))/(a*(D+l-x1)-h*x1-l*y1+D*(h-bcoord)+x1*bcoord)


def lim0_x21(y1,x1,bcoord,S,t):
    return [0,x21(y1,x1,bcoord)]

def lim0_x24(y1,x1,bcoord,S,t):
    return [0,x24(y1,x1,bcoord)]

def lim0_x22(y1,x1,bcoord,S,t):
    return [0,x22(y1,x1,bcoord)]

def limx21_x24(y1,x1,bcoord,S,t):
    return [x21(y1,x1,bcoord),x24(y1,x1,bcoord)]

def limx24_x21(y1,x1,bcoord,S,t):
    return [x24(y1,x1,bcoord),x21(y1,x1,bcoord)]

def limx21_x22(y1,x1,bcoord,S,t):
    return [x21(y1,x1,bcoord),x22(y1,x1,bcoord)]

def limx24_x22(y1,x1,bcoord,S,t):
    return [x24(y1,x1,bcoord),x22(y1,x1,bcoord)]

def limx23_x21(y1,x1,bcoord,S,t):
    return [x23(y1,x1,bcoord),x21(y1,x1,bcoord)]

def limx23_x24(y1,x1,bcoord,S,t):
    return [x23(y1,x1,bcoord),x24(y1,x1,bcoord)]

def lim0_x21_13(y1,x1,bcoord,S,t):
    return [0,x21_13(y1,x1,bcoord)]

def lim0_x24_13(y1,x1,bcoord,S,t):
    return [0,x24_13(y1,x1,bcoord)]

def lim0_x22_13(y1,x1,bcoord,S,t):
    return [0,x22_13(y1,x1,bcoord)]

def limx21_x24_13(y1,x1,bcoord,S,t):
    return [x21_13(y1,x1,bcoord),x24_13(y1,x1,bcoord)]

def limx24_x21_13(y1,x1,bcoord,S,t):
    return [x24_13(y1,x1,bcoord),x21_13(y1,x1,bcoord)]

def limx21_x22_13(y1,x1,bcoord,S,t):
    return [x21_13(y1,x1,bcoord),x22_13(y1,x1,bcoord)]

def limx24_x22_13(y1,x1,bcoord,S,t):
    return [x24_13(y1,x1,bcoord),x22_13(y1,x1,bcoord)]

def limx23_x21_13(y1,x1,bcoord,S,t):
    return [x23_13(y1,x1,bcoord),x21_13(y1,x1,bcoord)]

def limx23_x24_13(y1,x1,bcoord,S,t):
    return [x23_13(y1,x1,bcoord),x24_13(y1,x1,bcoord)]

def limx23a_D(y1,x1,bcoord,S,t):
    return [x23_a(y1,x1,bcoord),D]

def limx23b_D(y1,x1,bcoord,S,t):
    return [x23_b(y1,x1,bcoord),D]

def limx23d_D(y1,x1,bcoord,S,t):
    return [x23_d(y1,x1,bcoord),D]

def limx23e_D(y1,x1,bcoord,S,t):
    return [x23_e(y1,x1,bcoord),D]


def real_func(y2,x2,y1,x1,bcoord,S,t):
    N=(2*np.pi*j*hbar*2)/(m*(1+j*e))
    return scipy.real((N**(-S))*cmath.exp(((j*m*(1+j*e))/(2*hbar*t))*(K(x1,0)+K(x1,x2)+K(D+l,x2)+2*K(y1,y2/2)+0.5*K(0,y2)+K(bcoord,y2))))

def imag_func(y2,x2,y1,x1,bcoord,S,t):
    N=(2*np.pi*j*hbar*2)/(m*(1+j*e))
    return scipy.imag((N**(-S))*cmath.exp(((j*m*(1+j*e))/(2*hbar*t))*(K(x1,0)+K(x1,x2)+K(D+l,x2)+2*K(y1,y2/2)+0.5*K(0,y2)+K(bcoord,y2))))

def real_func3(y2,y1,x2,x1,bcoord,S,t):
    N=(2*np.pi*j*hbar*2)/(m*(1+j*e))
    return scipy.real((N**(-S))*cmath.exp(((j*m*(1+j*e))/(2*hbar*t))*(K(x1,0)+K(x1,x2)+K(D+l,x2)+2*K(y1,y2/2)+0.5*K(0,y2)+K(bcoord,y2))))

def imag_func3(y2,y1,x2,x1,bcoord,S,t):
    N=(2*np.pi*j*hbar*2)/(m*(1+j*e))
    return scipy.imag((N**(-S))*cmath.exp(((j*m*(1+j*e))/(2*hbar*t))*(K(x1,0)+K(x1,x2)+K(D+l,x2)+2*K(y1,y2/2)+0.5*K(0,y2)+K(bcoord,y2))))

def real_func1(y2,x2,x1,bcoord,S,t):
    N=(2*np.pi*j*hbar*t)/(m*(1+j*e))
    return scipy.real((1/np.sqrt(2))*(N**(-5/2))*cmath.exp(((j*m*(1+j*e))/(2*hbar*t))*(K(x1,0)+K(x1,x2)+K(D+l,x2)+0.5*K(y2,0)+K(y2,bcoord))))

def imag_func1(y2,x2,x1,bcoord,S,t):
    N=(2*np.pi*j*hbar*t)/(m*(1+j*e))
    return scipy.imag((1/np.sqrt(2))*(N**(-5/2))*cmath.exp(((j*m*(1+j*e))/(2*hbar*t))*(K(x1,0)+K(x1,x2)+K(D+l,x2)+0.5*K(y2,0)+K(y2,bcoord))))

def real_func5(x2,y1,x1,bcoord,S,t):
    N=(2*np.pi*j*hbar*2)/(m*(1+j*e))
    return scipy.real((1/np.sqrt(2))*(N**(-5/2))*cmath.exp(((j*m*(1+j*e))/(2*hbar*t))*(K(x1,0)+K(x1,x2)+K(D+l,x2)+K(y1,0)+0.5*K(y1,bcoord))))

def imag_func5(x2,y1,x1,bcoord,S,t):
    N=(2*np.pi*j*hbar*2)/(m*(1+j*e))
    return scipy.imag((1/np.sqrt(2))*(N**(-5/2))*cmath.exp(((j*m*(1+j*e))/(2*hbar*t))*(K(x1,0)+K(x1,x2)+K(D+l,x2)+K(y1,0)+0.5*K(y1,bcoord))))

def limy2_case1(x2,x1,bcoord,S,t):
    return [yinflim_case1(x2,bcoord),ysuplim_case1(x2,bcoord)]

def limy2_case1R(x2,x1,bcoord,S,t):
    return [-ysuplim_case1(x2,-bcoord),-yinflim_case1(x2,-bcoord)]

def limy2_case3(y1,x2,x1,bcoord,S,t):
    return [yinflim_case3(x2,y1,x1),ysuplim_case3(x2,y1,x1)]

def limy2_case3R(y1,x2,x1,bcoord,S,t):
    return [-ysuplim_case3(x2,-y1,x1),-yinflim_case3(x2,-y1,x1)]

def limy2_case5(x1,bcoord,S,t):
    return [yinflim_case5(x1),ysuplim_case5(x1)]

def limy2_case5R(x1,bcoord,S,t):
    return [-ysuplim_case5(x1),-yinflim_case5(x1)]

def limy2_case7inf1sup2(x2,y1,x1,bcoord,S,t):
    return [-ysuplim_case3(x2,-y1,x1),ysuplim_case1(x2,bcoord)]

def limy2_case7inf1sup1(x2,y1,x1,bcoord,S,t):
    return [-ysuplim_case3(x2,-y1,x1),-yinflim_case3(x2,-y1,x1)]

def limy2_case7inf2sup1(x2,y1,x1,bcoord,S,t):
    return [yinflim_case1(x2,bcoord),-yinflim_case3(x2,-y1,x1)]

def limy2_case7inf2sup2(x2,y1,x1,bcoord,S,t):
    return [yinflim_case1(x2,bcoord),ysuplim_case1(x2,bcoord)]

def limy2_case9inf2sup1(x2,y1,x1,bcoord,S,t):
    return [-ysuplim_case1(x2,-bcoord),-yinflim_case3(x2,-y1,x1)] 

def limy2_case9inf1sup2(x2,y1,x1,bcoord,S,t):
    return [-ysuplim_case3(x2,-y1,x1),-yinflim_case1(x2,-bcoord)]

def limy2_case9inf2sup2(x2,y1,x1,bcoord,S,t):
    return [-ysuplim_case1(x2,-bcoord),-yinflim_case1(x2,-bcoord)]

def limy2_case9Rinf2sup1(x2,y1,x1,bcoord,S,t):
    return [yinflim_case1(x2,-bcoord),ysuplim_case3(x2,-y1,x1)] 

def limy2_case9Rinf1sup2(x2,y1,x1,bcoord,S,t):
    return [yinflim_case3(x2,-y1,x1),ysuplim_case1(x2,-bcoord)]

def limy2_case9Rinf2sup2(x2,y1,x1,bcoord,S,t):
    return [yinflim_case1(x2,-bcoord),ysuplim_case1(x2,-bcoord)]

def limy2_case7Rinf1sup2(x2,y1,x1,bcoord,S,t):
    return [yinflim_case3(x2,-y1,x1),-yinflim_case1(x2,bcoord)]

def limy2_case7Rinf1sup1(x2,y1,x1,bcoord,S,t):
    return [yinflim_case3(x2,-y1,x1),ysuplim_case3(x2,-y1,x1)]

def limy2_case7Rinf2sup1(x2,y1,x1,bcoord,S,t):
    return [-ysuplim_case1(x2,bcoord),ysuplim_case3(x2,-y1,x1),]

def limy2_case7Rinf2sup2(x2,y1,x1,bcoord,S,t):
    return [-ysuplim_case1(x2,bcoord),-yinflim_case1(x2,bcoord)]

def limy2_case11inf2sup1(x2,y1,x1,bcoord,S,t):
    return [yinflim_case1(x2,bcoord),ysuplim_case3(x2,y1,x1)]

def limy2_case11inf2sup2(x2,y1,x1,bcoord,S,t):
    return [yinflim_case1(x2,bcoord),ysuplim_case1(x2,bcoord)]

def limy2_case11inf1sup2(x2,y1,x1,bcoord,S,t):
    return [yinflim_case3(x2,y1,x1),ysuplim_case1(x2,bcoord)]

def limy2_case11Rinf2sup1(x2,y1,x1,bcoord,S,t):
    return [-ysuplim_case1(x2,bcoord),-yinflim_case3(x2,y1,x1)]

def limy2_case11Rinf2sup2(x2,y1,x1,bcoord,S,t):
    return [-ysuplim_case1(x2,bcoord),-yinflim_case1(x2,bcoord)]

def limy2_case11Rinf1sup2(x2,y1,x1,bcoord,S,t):
    return [-ysuplim_case3(x2,y1,x1),-yinflim_case1(x2,bcoord)]

def limy2_case13inf1sup2(x2,y1,x1,bcoord,S,t):
    return [yinflim_case3(x2,y1,x1),-yinflim_case1(x2,-bcoord)]

def limy2_case13inf1sup1(x2,y1,x1,bcoord,S,t):
    return [yinflim_case3(x2,y1,x1),ysuplim_case3(x2,y1,x1)]

def limy2_case13inf2sup1(x2,y1,x1,bcoord,S,t):
    return [-ysuplim_case1(x2,-bcoord),ysuplim_case3(x2,y1,x1)]

def limy2_case13inf2sup2(x2,y1,x1,bcoord,S,t):
    return [-ysuplim_case1(x2,-bcoord),-yinflim_case1(x2,-bcoord)]

def limy2_case13Rinf1sup2(x2,y1,x1,bcoord,S,t):
    return [-ysuplim_case3(x2,y1,x1),ysuplim_case1(x2,-bcoord)]

def limy2_case13Rinf1sup1(x2,y1,x1,bcoord,S,t):
    return [-ysuplim_case3(x2,y1,x1),-yinflim_case3(x2,y1,x1)]

def limy2_case13Rinf2sup1(x2,y1,x1,bcoord,S,t):
    return [yinflim_case1(x2,-bcoord),-yinflim_case3(x2,y1,x1)]

def limy2_case13Rinf2sup2(x2,y1,x1,bcoord,S,t):
    return [yinflim_case1(x2,-bcoord),ysuplim_case1(x2,-bcoord)]

def int_Ybeforescreen1(bcoord): #S for number of steps + 1
    real_integral = nquad(real_func1, [limy2_case1,[0, D],[0,D]],args=(bcoord,3,2))
    imag_integral = nquad(imag_func1, [limy2_case1,[0, D],[0,D]],args=(bcoord,3,2))
    return real_integral[0] + 1j*imag_integral[0]
    
def int_Ybeforescreen1R(bcoord): #S for number of steps + 1
    real_integral = nquad(real_func1, [limy2_case1R,[0, D],[0,D]],args=(bcoord,3,2))
    imag_integral = nquad(imag_func1, [limy2_case1R,[0, D],[0,D]],args=(bcoord,3,2))
    return real_integral[0] + 1j*imag_integral[0]

    
def int_Ybeforescreen3(x2,x1,bcoord,S,t): #S for number of steps + 1
    if x1==D and x2==D:
        #print('cond1')
        real_integral1 = nquad(real_func3, [[a, a+h],[a, a+h]],args=(x2,x1,bcoord,3,2))
        imag_integral1 = nquad(imag_func3, [[a, a+h],[a, a+h]],args=(x2,x1,bcoord,3,2))
        return real_integral1[0] + 1j*imag_integral1[0]
    elif 0.49<x1<D and x2>D:
        #print('cond2',(x2,x1))
        real_integral1 = nquad(real_func3, [[-np.inf,np.inf],[a, a+h]],args=(x2,x1,bcoord,3,2))
        imag_integral1 = nquad(imag_func3, [[-np.inf,np.inf],[a, a+h]],args=(x2,x1,bcoord,3,2))
        return real_integral1[0] + 1j*imag_integral1[0]        
    else:
        #print('cond3',(x2,x1))
        real_integral2 = nquad(real_func3, [limy2_case3,[-np.inf, np.inf]],args=(x2,x1,bcoord,3,2))
        imag_integral2 = nquad(imag_func3, [limy2_case3,[-np.inf, np.inf]],args=(x2,x1,bcoord,3,2))
        return real_integral2[0] + 1j*imag_integral2[0]

def int_X3(bcoord):
    def realint3(x2,x1,bcoord,S,t):
        return scipy.real(int_Ybeforescreen3(x2,x1,bcoord,S,t))
    def imagint3(x2,x1,bcoord,S,t):
        return scipy.imag(int_Ybeforescreen3(x2,x1,bcoord,S,t))
    real_integral = nquad(realint3, [[D, D+l],[0,D]],args=(bcoord,3,2))
    imag_integral = nquad(imagint3, [[D, D+l],[0,D]],args=(bcoord,3,2))
    return real_integral[0] + 1j*imag_integral[0]
    
#print(int_Ybeforescreen3(D,D,0,3,2))
#print(int_X3(10))
def int_Ybeforescreen3R(x2,x1,bcoord,S,t): #S for number of steps + 1
    if x1==D and x2==D:
        print('cond1')
        real_integral1 = nquad(real_func3, [[-a-h, -a],[-a-h, -a]],args=(x2,x1,bcoord,3,2))
        imag_integral1 = nquad(imag_func3, [[-a-h, -a],[-a-h, -a]],args=(x2,x1,bcoord,3,2))
        return real_integral1[0] + 1j*imag_integral1[0]
    elif 0.49<x1<D and x2>D:
        print('cond2',(x2,x1))
        real_integral1 = nquad(real_func3, [[-np.inf,np.inf],[-a-h, -a]],args=(x2,x1,bcoord,3,2))
        imag_integral1 = nquad(imag_func3, [[-np.inf,np.inf],[-a-h, -a]],args=(x2,x1,bcoord,3,2))
        return real_integral1[0] + 1j*imag_integral1[0]        
    else:
        print('cond3',(x2,x1))
        real_integral2 = nquad(real_func3, [limy2_case3R,[-np.inf, np.inf]],args=(x2,x1,bcoord,3,2))
        imag_integral2 = nquad(imag_func3, [limy2_case3R,[-np.inf, np.inf]],args=(x2,x1,bcoord,3,2))
        return real_integral2[0] + 1j*imag_integral2[0]

def int_Yreal3(x2,x1,bcoord,S,t): #S for number of steps + 1
    if x1==D and x2==D:
        print('cond1')
        real_integral1 = nquad(real_func3, [[a, a+h],[a, a+h]],args=(x2,x1,bcoord,3,2))
        return real_integral1[0]
    elif 0.49<x1<D and x2>D:
        print('cond2',(x2,x1))
        real_integral1 = nquad(real_func3, [[-np.inf,np.inf],[a, a+h]],args=(x2,x1,bcoord,3,2))        
        return real_integral1[0]       
    else:
        print('cond3',(x2,x1))
        real_integral2 = nquad(real_func3, [limy2_case3,[-np.inf, np.inf]],args=(x2,x1,bcoord,3,2))
        return real_integral2[0]


#def case3_vegas(bcoord):
 #   def f(x):
 #       return int_Yreal3(x[1],x[0],bcoord,3,2)
  #  integ = vegas.Integrator([[0, D], [D, D+l]])
  #  result = integ(f, nitn=1, neval=1000)
   # return result


def int_Ybeforescreen5(bcoord): #S for number of steps + 1
    real_integral = nquad(real_func5, [[D, D+l],limy2_case5,[D,D+l]],args=(bcoord,3,2))
    imag_integral = nquad(imag_func5, [[D, D+l],limy2_case5,[D,D+l]],args=(bcoord,3,2))
    return real_integral[0] + 1j*imag_integral[0]

def int_Ybeforescreen5R(bcoord): #S for number of steps + 1
    real_integral = nquad(real_func5, [[D, D+l],limy2_case5R,[D,D+l]],args=(bcoord,3,2))
    imag_integral = nquad(imag_func5, [[D, D+l],limy2_case5R,[D,D+l]],args=(bcoord,3,2))
    return real_integral[0] + 1j*imag_integral[0]

def int2_7(y1,x1,bcoord,S,t): #S for number of steps + 1
    if 0<x21(y1,x1,bcoord)<D and  0<x22(y1,x1,bcoord)<D and x23(y1,x1,bcoord)<0 and 0<x24(y1,x1,bcoord)<D and  x21(y1,x1,bcoord)<x24(y1,x1,bcoord):
        real_integral = nquad(real_func, [limy2_case7inf1sup2,lim0_x21],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case7inf1sup1,limx21_x24],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case7inf2sup1,limx24_x22],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case7inf1sup2,lim0_x21],args=(y1,x1,bcoord,3,2))[0]+nquad(imag_func, [limy2_case7inf1sup1,limx21_x24],args=(y1,x1,bcoord,3,2))[0]+nquad(imag_func, [limy2_case7inf2sup1,limx24_x22],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif 0<x21(y1,x1,bcoord)<D and  0<x22(y1,x1,bcoord)<D and  x23(y1,x1,bcoord)<0 and 0<x24(y1,x1,bcoord)<D and  x21(y1,x1,bcoord)>x24(y1,x1,bcoord):
        real_integral = nquad(real_func, [limy2_case7inf1sup2,lim0_x24],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case7inf2sup2,limx24_x21],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case7inf2sup1,limx21_x22],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case7inf1sup2,lim0_x24],args=(y1,x1,bcoord,3,2))[0]+nquad(imag_func, [limy2_case7inf2sup2,limx24_x21],args=(y1,x1,bcoord,3,2))[0]+nquad(imag_func, [limy2_case7inf2sup1,limx21_x22],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif x21(y1,x1,bcoord)<0 and  0<x22(y1,x1,bcoord)<D and  x23(y1,x1,bcoord)<0 and 0<x24(y1,x1,bcoord)<D:
        real_integral = nquad(real_func, [limy2_case7inf1sup1,lim0_x24],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case7inf2sup1,limx24_x22],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case7inf1sup1,lim0_x24],args=(y1,x1,bcoord,3,2))[0]+nquad(imag_func, [limy2_case7inf2sup1,limx24_x22],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif 0<x21(y1,x1,bcoord)<D and  0<x22(y1,x1,bcoord)<D and  x23(y1,x1,bcoord)<0 and x24(y1,x1,bcoord)<0:
        real_integral = nquad(real_func, [limy2_case7inf2sup2,lim0_x21],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case7inf2sup1,limx21_x22],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case7inf2sup2,lim0_x21],args=(y1,x1,bcoord,3,2))[0]+nquad(imag_func, [limy2_case7inf2sup1,limx21_x22],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif x21(y1,x1,bcoord)<0 and  0<x22(y1,x1,bcoord)<D and  x23(y1,x1,bcoord)<0 and x24(y1,x1,bcoord)<0:
        real_integral = nquad(real_func, [limy2_case7inf2sup1,lim0_x22],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case7inf2sup1,lim0_x22],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif 0<x21(y1,x1,bcoord)<D and  0<x22(y1,x1,bcoord)<D and  0<x23(y1,x1,bcoord)<D and 0<x24(y1,x1,bcoord)<D and x24(y1,x1,bcoord)>x21(y1,x1,bcoord):
        real_integral = nquad(real_func, [limy2_case7inf1sup2,limx23_x21],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case7inf1sup1,limx21_x24],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case7inf2sup1,limx24_x22],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case7inf1sup2,limx23_x21],args=(y1,x1,bcoord,3,2))[0]+nquad(imag_func, [limy2_case7inf1sup1,limx21_x24],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case7inf2sup1,limx24_x22],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif 0<x21(y1,x1,bcoord)<D and  0<x22(y1,x1,bcoord)<D and  0<x23(y1,x1,bcoord)<D and 0<x24(y1,x1,bcoord)<D and x24(y1,x1,bcoord)<x21(y1,x1,bcoord):
        real_integral = nquad(real_func, [limy2_case7inf1sup2,limx23_x24],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case7inf2sup2,limx24_x21],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case7inf2sup1,limx21_x22],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case7inf1sup2,limx23_x24],args=(y1,x1,bcoord,3,2))[0]+nquad(imag_func, [limy2_case7inf2sup2,limx24_x21],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case7inf2sup1,limx21_x22],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral 
    
    else:
        return 0

def int2_7R(y1,x1,bcoord,S,t): #S for number of steps + 1
    if 0<x21(-y1,x1,-bcoord)<D and  0<x22(-y1,x1,-bcoord)<D and x23(-y1,x1,-bcoord)<0 and 0<x24(-y1,x1,-bcoord)<D and  x21(-y1,x1,-bcoord)<x24(-y1,x1,-bcoord):
        real_integral = nquad(real_func, [limy2_case7Rinf2sup1,lim0_x21],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case7Rinf1sup1,limx21_x24],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case7Rinf1sup2,limx24_x22],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case7Rinf2sup1,lim0_x21],args=(-y1,x1,-bcoord,3,2))[0]+nquad(imag_func, [limy2_case7Rinf1sup1,limx21_x24],args=(-y1,x1,-bcoord,3,2))[0]+nquad(imag_func, [limy2_case7Rinf1sup2,limx24_x22],args=(-y1,x1,-bcoord,3,2))[0]
        print('condic1')
        return real_integral + 1j*imag_integral
    
    elif 0<x21(-y1,x1,-bcoord)<D and  0<x22(-y1,x1,-bcoord)<D and  x23(-y1,x1,-bcoord)<0 and 0<x24(-y1,x1,-bcoord)<D and  x21(-y1,x1,-bcoord)>x24(-y1,x1,-bcoord):
        real_integral = nquad(real_func, [limy2_case7Rinf2sup1,lim0_x24],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case7Rinf2sup2,limx24_x21],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case7Rinf1sup2,limx21_x22],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case7Rinf2sup1,lim0_x24],args=(-y1,x1,-bcoord,3,2))[0]+nquad(imag_func, [limy2_case7Rinf2sup2,limx24_x21],args=(-y1,x1,-bcoord,3,2))[0]+nquad(imag_func, [limy2_case7Rinf1sup2,limx21_x22],args=(-y1,x1,-bcoord,3,2))[0]
        print('condic2')
        return real_integral + 1j*imag_integral
    
    elif x21(-y1,x1,-bcoord)<0 and  0<x22(-y1,x1,-bcoord)<D and  x23(-y1,x1,-bcoord)<0 and 0<x24(-y1,x1,-bcoord)<D:
        real_integral = nquad(real_func, [limy2_case7Rinf1sup1,lim0_x24],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case7Rinf1sup2,limx24_x22],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case7Rinf1sup1,lim0_x24],args=(-y1,x1,-bcoord,3,2))[0]+nquad(imag_func, [limy2_case7Rinf1sup2,limx24_x22],args=(-y1,x1,-bcoord,3,2))[0]
        print('condic3')
        return real_integral + 1j*imag_integral
    
    elif 0<x21(-y1,x1,-bcoord)<D and  0<x22(-y1,x1,-bcoord)<D and  x23(-y1,x1,-bcoord)<0 and x24(-y1,x1,-bcoord)<0:
        real_integral = nquad(real_func, [limy2_case7Rinf2sup2,lim0_x21],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case7Rinf1sup2,limx21_x22],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case7Rinf2sup2,lim0_x21],args=(-y1,x1,-bcoord,3,2))[0]+nquad(imag_func, [limy2_case7Rinf1sup2,limx21_x22],args=(-y1,x1,-bcoord,3,2))[0]
        print('condic4')
        return real_integral + 1j*imag_integral
    
    elif x21(-y1,x1,-bcoord)<0 and  0<x22(-y1,x1,-bcoord)<D and  x23(-y1,x1,-bcoord)<0 and x24(-y1,x1,-bcoord)<0:
        real_integral = nquad(real_func, [limy2_case7Rinf1sup2,lim0_x22],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case7Rinf1sup2,lim0_x22],args=(-y1,x1,-bcoord,3,2))[0]
        print('condic5')
        return real_integral + 1j*imag_integral
    
    elif 0<x21(-y1,x1,-bcoord)<D and  0<x22(-y1,x1,-bcoord)<D and  0<x23(-y1,x1,-bcoord)<D and 0<x24(-y1,x1,-bcoord)<D and x24(-y1,x1,-bcoord)>x21(-y1,x1,-bcoord):
        real_integral = nquad(real_func, [limy2_case7Rinf2sup1,limx23_x21],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case7Rinf1sup1,limx21_x24],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case7Rinf1sup2,limx24_x22],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case7Rinf2sup1,limx23_x21],args=(-y1,x1,-bcoord,3,2))[0]+nquad(imag_func, [limy2_case7Rinf1sup1,limx21_x24],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case7Rinf1sup2,limx24_x22],args=(-y1,x1,-bcoord,3,2))[0]
        print('condic6')
        return real_integral + 1j*imag_integral
    
    elif 0<x21(-y1,x1,-bcoord)<D and  0<x22(-y1,x1,-bcoord)<D and  0<x23(-y1,x1,-bcoord)<D and 0<x24(-y1,x1,-bcoord)<D and x24(-y1,x1,-bcoord)<x21(-y1,x1,-bcoord):
        real_integral = nquad(real_func, [limy2_case7Rinf2sup1,limx23_x24],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case7Rinf2sup2,limx24_x21],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case7Rinf1sup2,limx21_x22],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case7Rinf2sup1,limx23_x24],args=(-y1,x1,-bcoord,3,2))[0]+nquad(imag_func, [limy2_case7Rinf2sup2,limx24_x21],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case7Rinf1sup2,limx21_x22],args=(-y1,x1,-bcoord,3,2))[0]
        print('condic7')
        return real_integral + 1j*imag_integral 
    
    else:
        print('cero')
        return 0

def int2_9(y1,x1,bcoord,S,t): #S for number of steps + 1
    if 0<x23_a(y1,x1,bcoord)<D:
        real_integral = nquad(real_func, [limy2_case9inf2sup1,limx23a_D],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case9inf2sup1,limx23a_D],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif 0<x23_b(y1,x1,bcoord)<D:
        real_integral = nquad(real_func, [limy2_case9inf1sup2,limx23b_D],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case9inf1sup2,limx23b_D],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif -ysuplim_case3(0,-y1,x1)<-ysuplim_case1(0,-bcoord) and -yinflim_case3(0,-y1,x1)<-yinflim_case1(0,-bcoord):
        real_integral = nquad(real_func, [limy2_case9inf2sup1,[0,D]],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case9inf2sup1,[0,D]],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif -ysuplim_case3(0,-y1,x1)<-ysuplim_case1(0,-bcoord) and -yinflim_case3(0,-y1,x1)>-yinflim_case1(0,-bcoord):
        real_integral = nquad(real_func, [limy2_case9inf2sup2,[0,D]],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case9inf2sup2,[0,D]],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif -ysuplim_case3(0,-y1,x1)>-ysuplim_case1(0,-bcoord) and -yinflim_case3(0,-y1,x1)>-yinflim_case1(0,-bcoord):
        real_integral = nquad(real_func, [limy2_case9inf1sup2,[0,D]],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case9inf1sup2,[0,D]],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    else:
       return 0
   
def int2_9R(y1,x1,bcoord,S,t): #S for number of steps + 1
    if 0<x23_a(-y1,x1,-bcoord)<D:
        real_integral = nquad(real_func, [limy2_case9Rinf1sup2,limx23a_D],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case9Rinf1sup2,limx23a_D],args=(-y1,x1,-bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif 0<x23_b(-y1,x1,-bcoord)<D:
        real_integral = nquad(real_func, [limy2_case9Rinf2sup1,limx23b_D],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case9Rinf2sup1,limx23b_D],args=(-y1,x1,-bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif ysuplim_case3(0,y1,x1)<ysuplim_case1(0,bcoord) and yinflim_case3(0,y1,x1)<yinflim_case1(0,bcoord):
        real_integral = nquad(real_func, [limy2_case9Rinf2sup1,[0,D]],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case9Rinf2sup1,[0,D]],args=(-y1,x1,-bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif ysuplim_case3(0,y1,x1)>ysuplim_case1(0,bcoord) and yinflim_case3(0,y1,x1)<yinflim_case1(0,bcoord):
        real_integral = nquad(real_func, [limy2_case9Rinf2sup2,[0,D]],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case9Rinf2sup2,[0,D]],args=(-y1,x1,-bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif ysuplim_case3(0,y1,x1)>ysuplim_case1(0,bcoord) and yinflim_case3(0,y1,x1)>yinflim_case1(0,bcoord):
        real_integral = nquad(real_func, [limy2_case9Rinf1sup2,[0,D]],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case9Rinf1sup2,[0,D]],args=(-y1,x1,-bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    else:
       return 0

def int2_11(y1,x1,bcoord,S,t): #S for number of steps + 1
    if 0<=x23_d(y1,x1,bcoord)<=D:
        real_integral = nquad(real_func, [limy2_case11inf2sup1,limx23d_D],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case11inf2sup1,limx23d_D],args=(y1,x1,bcoord,3,2))[0]
        print('x23_d')
        return real_integral + 1j*imag_integral
    
    elif 0<=x23_e(y1,x1,bcoord)<=D:
        real_integral = nquad(real_func, [limy2_case11inf1sup2,limx23e_D],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case11inf1sup2,limx23e_D],args=(y1,x1,bcoord,3,2))[0]
        print('x23_e')
        return real_integral + 1j*imag_integral
    
    elif ysuplim_case3(0,y1,x1)<ysuplim_case1(0,bcoord) and yinflim_case3(0,y1,x1)<yinflim_case1(0,bcoord):
        real_integral = nquad(real_func, [limy2_case11inf2sup1,[0,D]],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case11inf2sup1,[0,D]],args=(y1,x1,bcoord,3,2))[0]
        print('cono 2 mayor que 1')
        return real_integral + 1j*imag_integral
    
    elif ysuplim_case3(0,y1,x1)>ysuplim_case1(0,bcoord) and yinflim_case3(0,y1,x1)<yinflim_case1(0,bcoord):
        real_integral = nquad(real_func, [limy2_case11inf2sup2,[0,D]],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case11inf2sup2,[0,D]],args=(y1,x1,bcoord,3,2))[0]
        print('cono 2 dentro de 1')
        return real_integral + 1j*imag_integral
    
    elif ysuplim_case3(0,y1,x1)>ysuplim_case1(0,bcoord) and yinflim_case3(0,y1,x1)>yinflim_case1(0,bcoord):
        real_integral = nquad(real_func, [limy2_case11inf1sup2,[0,D]],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case11inf1sup2,[0,D]],args=(y1,x1,bcoord,3,2))[0]
        print('cono 1 mayor que 2')
        return real_integral + 1j*imag_integral
    
    else:
        print('cero')
        return 0

def int2_11R(y1,x1,bcoord,S,t): #S for number of steps + 1
    if 0<=x23_d(-y1,x1,-bcoord)<=D:
        real_integral = nquad(real_func, [limy2_case11Rinf1sup2,limx23d_D],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case11Rinf1sup2,limx23d_D],args=(-y1,x1,-bcoord,3,2))[0]
        print('x23_d')
        return real_integral + 1j*imag_integral
    
    elif 0<=x23_e(-y1,x1,-bcoord)<=D:
        real_integral = nquad(real_func, [limy2_case11Rinf2sup1,limx23e_D],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case11Rinf2sup1,limx23e_D],args=(-y1,x1,-bcoord,3,2))[0]
        print('x23_e')
        return real_integral + 1j*imag_integral
    
    elif -ysuplim_case3(0,-y1,x1)>-ysuplim_case1(0,-bcoord) and -yinflim_case3(0,-y1,x1)>-yinflim_case1(0,-bcoord):
        real_integral = nquad(real_func, [limy2_case11Rinf1sup2,[0,D]],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case11Rinf1sup2,[0,D]],args=(-y1,x1,-bcoord,3,2))[0]
        print('cono 1 mayor que 2')
        return real_integral + 1j*imag_integral
    
    elif -ysuplim_case3(0,-y1,x1)<-ysuplim_case1(0,-bcoord) and -yinflim_case3(0,-y1,x1)>-yinflim_case1(0,-bcoord):
        real_integral = nquad(real_func, [limy2_case11Rinf2sup2,[0,D]],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case11Rinf2sup2,[0,D]],args=(-y1,x1,-bcoord,3,2))[0]
        print('cono 2 dentro de 1')
        return real_integral + 1j*imag_integral
    
    elif -ysuplim_case3(0,-y1,x1)<-ysuplim_case1(0,-bcoord) and -yinflim_case3(0,-y1,x1)<-yinflim_case1(0,-bcoord):
        real_integral = nquad(real_func, [limy2_case11Rinf2sup1,[0,D]],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case11Rinf2sup1,[0,D]],args=(-y1,x1,-bcoord,3,2))[0]
        print('cono 2 mayor que 1')
        return real_integral + 1j*imag_integral
    
    else:
        print('cero')
        return 0
        
def int2_13(y1,x1,bcoord,S,t): #S for number of steps + 1
    if 0<x21_13(y1,x1,bcoord)<D and  0<x22_13(y1,x1,bcoord)<D and x23_13(y1,x1,bcoord)<0 and 0<x24_13(y1,x1,bcoord)<D and  x21_13(y1,x1,bcoord)<x24_13(y1,x1,bcoord):
        real_integral = nquad(real_func, [limy2_case13inf2sup1,lim0_x21_13],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case13inf2sup2,limx21_x24_13],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case13inf1sup2,limx24_x22_13],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case13inf2sup1,lim0_x21_13],args=(y1,x1,bcoord,3,2))[0]+nquad(imag_func, [limy2_case13inf2sup2,limx21_x24_13],args=(y1,x1,bcoord,3,2))[0]+nquad(imag_func, [limy2_case13inf1sup2,limx24_x22_13],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif 0<x21_13(y1,x1,bcoord)<D and  0<x22_13(y1,x1,bcoord)<D and  x23_13(y1,x1,bcoord)<0 and 0<x24_13(y1,x1,bcoord)<D and  x21_13(y1,x1,bcoord)>x24_13(y1,x1,bcoord):
        real_integral = nquad(real_func, [limy2_case13inf2sup1,lim0_x24_13],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case13inf1sup1,limx24_x21_13],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case13inf1sup2,limx21_x22_13],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case13inf2sup1,lim0_x24_13],args=(y1,x1,bcoord,3,2))[0]+nquad(imag_func, [limy2_case13inf1sup1,limx24_x21_13],args=(y1,x1,bcoord,3,2))[0]+nquad(imag_func, [limy2_case13inf1sup2,limx21_x22_13],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif x21_13(y1,x1,bcoord)<0 and  0<x22_13(y1,x1,bcoord)<D and  x23_13(y1,x1,bcoord)<0 and 0<x24_13(y1,x1,bcoord)<D:
        real_integral = nquad(real_func, [limy2_case13inf2sup2,lim0_x24_13],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case13inf1sup2,limx24_x22_13],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case13inf2sup2,lim0_x24_13],args=(y1,x1,bcoord,3,2))[0]+nquad(imag_func, [limy2_case13inf1sup2,limx24_x22_13],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif 0<x21_13(y1,x1,bcoord)<D and  0<x22_13(y1,x1,bcoord)<D and  x23_13(y1,x1,bcoord)<0 and x24_13(y1,x1,bcoord)<0:
        real_integral = nquad(real_func, [limy2_case13inf1sup1,lim0_x21_13],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case13inf1sup2,limx21_x22_13],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case13inf1sup1,lim0_x21_13],args=(y1,x1,bcoord,3,2))[0]+nquad(imag_func, [limy2_case13inf1sup2,limx21_x22_13],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif x21_13(y1,x1,bcoord)<0 and  0<x22_13(y1,x1,bcoord)<D and  x23_13(y1,x1,bcoord)<0 and x24_13(y1,x1,bcoord)<0:
        real_integral = nquad(real_func, [limy2_case13inf1sup2,lim0_x22_13],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case13inf1sup2,lim0_x22_13],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif 0<x21_13(y1,x1,bcoord)<D and  0<x22_13(y1,x1,bcoord)<D and  0<x23_13(y1,x1,bcoord)<D and 0<x24_13(y1,x1,bcoord)<D and x24_13(y1,x1,bcoord)>x21_13(y1,x1,bcoord):
        real_integral = nquad(real_func, [limy2_case13inf2sup1,limx23_x21_13],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case13inf2sup2,limx21_x24_13],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case13inf1sup2,limx24_x22_13],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case13inf2sup1,limx23_x21_13],args=(y1,x1,bcoord,3,2))[0]+nquad(imag_func, [limy2_case13inf2sup2,limx21_x24_13],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case13inf1sup2,limx24_x22_13],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral
    
    elif 0<x21_13(y1,x1,bcoord)<D and  0<x22_13(y1,x1,bcoord)<D and  0<x23_13(y1,x1,bcoord)<D and 0<x24_13(y1,x1,bcoord)<D and x24_13(y1,x1,bcoord)<x21_13(y1,x1,bcoord):
        real_integral = nquad(real_func, [limy2_case13inf2sup1,limx23_x24_13],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case13inf1sup1,limx24_x21_13],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case13inf1sup2,limx21_x22_13],args=(y1,x1,bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case13inf2sup1,limx23_x24_13],args=(y1,x1,bcoord,3,2))[0]+nquad(imag_func, [limy2_case13inf1sup1,limx24_x21_13],args=(y1,x1,bcoord,3,2))[0]+nquad(real_func, [limy2_case13inf1sup2,limx21_x22_13],args=(y1,x1,bcoord,3,2))[0]
        return real_integral + 1j*imag_integral 
    
    else:
        return 0
    
def int2_13R(y1,x1,bcoord,S,t): #S for number of steps + 1
    if 0<x21_13(-y1,x1,-bcoord)<D and  0<x22_13(-y1,x1,-bcoord)<D and x23_13(-y1,x1,-bcoord)<0 and 0<x24_13(-y1,x1,-bcoord)<D and  x21_13(-y1,x1,-bcoord)<x24_13(-y1,x1,-bcoord):
        real_integral = nquad(real_func, [limy2_case13Rinf1sup2,lim0_x21_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case13Rinf2sup2,limx21_x24_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case13Rinf2sup1,limx24_x22_13],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case13Rinf1sup2,lim0_x21_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(imag_func, [limy2_case13Rinf2sup2,limx21_x24_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(imag_func, [limy2_case13Rinf2sup1,limx24_x22_13],args=(-y1,x1,-bcoord,3,2))[0]
        print('kk')
        return real_integral + 1j*imag_integral
    
    elif 0<x21_13(-y1,x1,-bcoord)<D and  0<x22_13(-y1,x1,-bcoord)<D and  x23_13(-y1,x1,-bcoord)<0 and 0<x24_13(-y1,x1,-bcoord)<D and  x21_13(-y1,x1,-bcoord)>x24_13(-y1,x1,-bcoord):
        real_integral = nquad(real_func, [limy2_case13Rinf1sup2,lim0_x24_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case13Rinf1sup1,limx24_x21_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case13Rinf2sup1,limx21_x22_13],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case13Rinf1sup2,lim0_x24_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(imag_func, [limy2_case13Rinf1sup1,limx24_x21_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(imag_func, [limy2_case13Rinf2sup1,limx21_x22_13],args=(-y1,x1,-bcoord,3,2))[0]
        print('kk')
        return real_integral + 1j*imag_integral
    
    elif x21_13(-y1,x1,-bcoord)<0 and  0<x22_13(-y1,x1,-bcoord)<D and  x23_13(-y1,x1,-bcoord)<0 and 0<x24_13(-y1,x1,-bcoord)<D:
        real_integral = nquad(real_func, [limy2_case13Rinf2sup2,lim0_x24_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case13Rinf2sup1,limx24_x22_13],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case13Rinf2sup2,lim0_x24_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(imag_func, [limy2_case13Rinf2sup1,limx24_x22_13],args=(-y1,x1,-bcoord,3,2))[0]
        print('kk')
        return real_integral + 1j*imag_integral
    
    elif 0<x21_13(-y1,x1,-bcoord)<D and  0<x22_13(-y1,x1,-bcoord)<D and  x23_13(-y1,x1,-bcoord)<0 and x24_13(-y1,x1,-bcoord)<0:
        real_integral = nquad(real_func, [limy2_case13Rinf1sup1,lim0_x21_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case13Rinf2sup1,limx21_x22_13],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case13Rinf1sup1,lim0_x21_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(imag_func, [limy2_case13Rinf2sup1,limx21_x22_13],args=(-y1,x1,-bcoord,3,2))[0]
        print('kk')
        return real_integral + 1j*imag_integral
    
    elif x21_13(-y1,x1,-bcoord)<0 and  0<x22_13(-y1,x1,-bcoord)<D and  x23_13(-y1,x1,-bcoord)<0 and x24_13(-y1,x1,-bcoord)<0:
        real_integral = nquad(real_func, [limy2_case13Rinf2sup1,lim0_x22_13],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case13Rinf2sup1,lim0_x22_13],args=(-y1,x1,-bcoord,3,2))[0]
        print('kk')
        return real_integral + 1j*imag_integral
    
    elif 0<x21_13(-y1,x1,-bcoord)<D and  0<x22_13(-y1,x1,-bcoord)<D and  0<x23_13(-y1,x1,-bcoord)<D and 0<x24_13(-y1,x1,-bcoord)<D and x24_13(-y1,x1,-bcoord)>x21_13(-y1,x1,-bcoord):
        real_integral = nquad(real_func, [limy2_case13Rinf1sup2,limx23_x21_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case13Rinf2sup2,limx21_x24_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case13Rinf2sup1,limx24_x22_13],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case13Rinf1sup2,limx23_x21_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(imag_func, [limy2_case13Rinf2sup2,limx21_x24_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case13Rinf2sup1,limx24_x22_13],args=(-y1,x1,-bcoord,3,2))[0]
        print('kk')
        return real_integral + 1j*imag_integral
    
    elif 0<x21_13(-y1,x1,-bcoord)<D and  0<x22_13(-y1,x1,-bcoord)<D and  0<x23_13(-y1,x1,-bcoord)<D and 0<x24_13(-y1,x1,-bcoord)<D and x24_13(-y1,x1,-bcoord)<x21_13(-y1,x1,-bcoord):
        real_integral = nquad(real_func, [limy2_case13Rinf1sup2,limx23_x24_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case13Rinf1sup1,limx24_x21_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case13Rinf2sup1,limx21_x22_13],args=(-y1,x1,-bcoord,3,2))[0]
        imag_integral = nquad(imag_func, [limy2_case13Rinf1sup2,limx23_x24_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(imag_func, [limy2_case13Rinf1sup1,limx24_x21_13],args=(-y1,x1,-bcoord,3,2))[0]+nquad(real_func, [limy2_case13Rinf2sup1,limx21_x22_13],args=(-y1,x1,-bcoord,3,2))[0]
        print('kk')
        return real_integral + 1j*imag_integral 
    
    else:
        print('zero')
        return 0
    
def int_Ybeforescreen7(bcoord): #S for number of steps + 1
    def realint2_7(y1,x1,bcoord,S,t):
        return scipy.real(int2_7(y1,x1,bcoord,S,t))
    def imagint2_7(y1,x1,bcoord,S,t):
        return scipy.imag(int2_7(y1,x1,bcoord,S,t))
    real_integral = nquad(realint2_7, [limy2_case5,[D, D+l]],args=(bcoord,3,2))
    imag_integral = nquad(imagint2_7, [limy2_case5,[D, D+l]],args=(bcoord,3,2))
    return real_integral[0] + 1j*imag_integral[0]

def int_Ybeforescreen7R(bcoord): #S for number of steps + 1
    def realint2_7R(y1,x1,bcoord,S,t):
        return scipy.real(int2_7R(y1,x1,bcoord,S,t))
    def imagint2_7R(y1,x1,bcoord,S,t):
        return scipy.imag(int2_7R(y1,x1,bcoord,S,t))
    real_integral = nquad(realint2_7R, [limy2_case5R,[D, D+l]],args=(bcoord,3,2))
    imag_integral = nquad(imagint2_7R, [limy2_case5R,[D, D+l]],args=(bcoord,3,2))
    return real_integral[0] + 1j*imag_integral[0]

def int_Ybeforescreen9(bcoord): #S for number of steps + 1
    def realint2_9(y1,x1,bcoord,S,t):
        return scipy.real(int2_9(y1,x1,bcoord,S,t))
    def imagint2_9(y1,x1,bcoord,S,t):
        return scipy.imag(int2_9(y1,x1,bcoord,S,t))
    real_integral = nquad(realint2_9, [limy2_case5,[D, D+l]],args=(bcoord,3,2))
    imag_integral = nquad(imagint2_9, [limy2_case5,[D, D+l]],args=(bcoord,3,2))
    return real_integral[0] + 1j*imag_integral[0]

def int_Ybeforescreen9R(bcoord): #S for number of steps + 1
    def realint2_9R(y1,x1,bcoord,S,t):
        return scipy.real(int2_9R(y1,x1,bcoord,S,t))
    def imagint2_9R(y1,x1,bcoord,S,t):
        return scipy.imag(int2_9R(y1,x1,bcoord,S,t))
    real_integral = nquad(realint2_9R, [limy2_case5R,[D, D+l]],args=(bcoord,3,2))
    imag_integral = nquad(imagint2_9R, [limy2_case5R,[D, D+l]],args=(bcoord,3,2))
    return real_integral[0] + 1j*imag_integral[0]

def int_Ybeforescreen11(bcoord): #S for number of steps + 1
    def realint2_11(y1,x1,bcoord,S,t):
        return scipy.real(int2_11(y1,x1,bcoord,S,t))
    def imagint2_11(y1,x1,bcoord,S,t):
        return scipy.imag(int2_11(y1,x1,bcoord,S,t))
    real_integral = nquad(realint2_11, [limy2_case5,[D, D+l]],args=(bcoord,3,2))
    imag_integral = nquad(imagint2_11, [limy2_case5,[D, D+l]],args=(bcoord,3,2))
    return real_integral[0] + 1j*imag_integral[0]

def int_Ybeforescreen11R(bcoord): #S for number of steps + 1
    def realint2_11R(y1,x1,bcoord,S,t):
        return scipy.real(int2_11R(y1,x1,bcoord,S,t))
    def imagint2_11R(y1,x1,bcoord,S,t):
        return scipy.imag(int2_11R(y1,x1,bcoord,S,t))
    real_integral = nquad(realint2_11R, [limy2_case5R,[D, D+l]],args=(bcoord,3,2))
    imag_integral = nquad(imagint2_11R, [limy2_case5R,[D, D+l]],args=(bcoord,3,2))
    return real_integral[0] + 1j*imag_integral[0]

def int_Ybeforescreen13(bcoord): #S for number of steps + 1
    def realint2_13(y1,x1,bcoord,S,t):
        return scipy.real(int2_13(y1,x1,bcoord,S,t))
    def imagint2_13(y1,x1,bcoord,S,t):
        return scipy.imag(int2_13(y1,x1,bcoord,S,t))
    real_integral = nquad(realint2_13, [limy2_case5,[D, D+l]],args=(bcoord,3,2))
    imag_integral = nquad(imagint2_13, [limy2_case5,[D, D+l]],args=(bcoord,3,2))
    return real_integral[0] + 1j*imag_integral[0]

def int_Ybeforescreen13R(bcoord): #S for number of steps + 1
    def realint2_13R(y1,x1,bcoord,S,t):
        return scipy.real(int2_13R(y1,x1,bcoord,S,t))
    def imagint2_13R(y1,x1,bcoord,S,t):
        return scipy.imag(int2_13R(y1,x1,bcoord,S,t))
    real_integral = nquad(realint2_13R, [limy2_case5R,[D, D+l]],args=(bcoord,3,2))
    imag_integral = nquad(imagint2_13R, [limy2_case5R,[D, D+l]],args=(bcoord,3,2))
    return real_integral[0] + 1j*imag_integral[0]

def sumfuncY3(lista,bcoord):
    aux2=[]
    for i in lista:
        y=int_Ybeforescreen3(i[1],i[0],bcoord,3,2)
        aux2.append(y)
        print(y,len(aux2),bcoord)
    return sum(aux2)

def sumfuncY3R(lista,bcoord):
    aux2=[]
    for i in lista:
        y=int_Ybeforescreen3R(i[1],i[0],bcoord,3,2)
        aux2.append(y)
        print(y,len(aux2),bcoord)
    return sum(aux2)
    
def int_montecarlo3(bcoord):
    x12 =[(random.uniform (0,D),random.uniform (D,D+l)) for k in range(1000)]
    meanfunct=sumfuncY3(x12,bcoord)/1000
    I2=D*l*(meanfunct)
    return(I2)
    
def int_montecarlo3R(bcoord):
    x12 =[(random.uniform (0,D),random.uniform (D,D+l)) for k in range(1000)]
    meanfunct=sumfuncY3R(x12,bcoord)/1000
    I2=D*l*(meanfunct)
    return(I2)
    

def freal(y2):
    return real_func1(y2,0.5,0.5,9,3,2)

def fimag(y2):
    return imag_func1(y2,0.5,0.5,9,3,2)

def plotinflim1(x):
    return -ysuplim_case1(x,-1.5)

def plotsuplim1(x):
    return -yinflim_case1(x,-1.5)

def plotinflim3(x):
    return yinflim_case3(x,1.8e-3,0.6)

def plotsuplim3(x):
    return ysuplim_case3(x,1.8e-3,0.6)

def int_Xfixed3(bcoord):
    return int_Ybeforescreen3(D+l/2,D/2,bcoord,3,2)

def int_Xfixed3R(bcoord):
    return int_Ybeforescreen3R(D+l/2,D/2,bcoord,3,2)

#r=Pool(5)
#yb=np.arange(-20,20,1e-1)
#x2=np.arange(0,1+1e-1,1e-1)
#case1inflim=np.array(r.map(plotinflim1,x2))
#case1suplim=np.array(r.map(plotsuplim1,x2))
#case3inflim=np.array(r.map(plotinflim3,x2))
#case3suplim=np.array(r.map(plotsuplim3,x2))
#imag=np.array(r.map(fimag,y2))
#k=Pool(10)
#kernel=np.array(k.map(int_Ybeforescreen13R,yb))
#kernel3=np.array(k.map(int_montecarlo3R,yb))
#pbb=abs(kernel)**2
#plt.title("Probability for two step case 14")
#plt.xlabel("Screen coords(meters)")
#plt.ylabel("Probability")
#plt.plot(yb,pbb,'bo')
#plt.plot(x2,case1inflim,color='red')
#plt.plot(x2,case1suplim,color='red')
#plt.plot(x2,case3inflim,color='blue')
#plt.plot(x2,case3suplim,color='blue')
#plt.axvline(x=0.5)
#plt.axvline(x=x23(-1.8e-3,0.6,-1.5),color='purple')
#plt.show()
#plt.savefig("2step_case13R400.png")
#np.savetxt('2stepvalues_case13R400',kernel,delimiter=',')
print("--- %s seconds ---" % (time.time() - start_time))



