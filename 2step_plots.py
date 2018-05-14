import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import os
os.chdir('/Users/tblantano/Desktop/plots&values')
l1,l1R,l3,l3R,l5,l5R,l7,l7R,l9,l9R,l11,l11R,l13,l13R=[],[],[],[],[],[],[],[],[],[],[],[],[],[]
kernel1,kernel1R,kernel3,kernel3R,kernel5,kernel5R,kernel7,kernel7R,kernel9,kernel9R,kernel11,kernel11R,kernel13,kernel13R=[],[],[],[],[],[],[],[],[],[],[],[],[],[]
f1 = open('2stepvalues_case1400','r')
f1R = open('2stepvalues_case1R400','r')
f3= open('2stepvalues_case3400','r')
f3R= open('2stepvalues_case3RMC400','r')
f5 = open('2stepvalues_case5400','r')
f5R = open('2stepvalues_case5R400','r')
f7 = open('2stepvalues_case7400','r')
f7R = open('2stepvalues_case7R400','r')
f9 = open('2stepvalues_case9400','r')
f9R = open('2stepvalues_case9R400','r')
f11 = open('2stepvalues_case11400','r')
f11R = open('2stepvalues_case11R400','r')
f13 = open('2stepvalues_case13400','r')
f13R = open('2stepvalues_case13R400','r')

for line in f1:
    l1.append(line.strip())
for line in f1R:
    l1R.append(line.strip())
for line in f5:
    l5.append(line.strip())
for line in f3:
    l3.append(line.strip())
for line in f3R:
    l3R.append(line.strip())
for line in f5R:
    l5R.append(line.strip())
for line in f7:
    l7.append(line.strip())
for line in f7R:
    l7R.append(line.strip())
for line in f9:
    l9.append(line.strip())
for line in f9R:
    l9R.append(line.strip())
for line in f11:
    l11.append(line.strip())
for line in f11R:
    l11R.append(line.strip())
for line in f13:
    l13.append(line.strip())
for line in f13R:
    l13R.append(line.strip())

[kernel1.append(complex(i.replace('+-','-'))) for i in l1]
[kernel1R.append(complex(i.replace('+-','-'))) for i in l1R]
[kernel3.append(complex(i.replace('+-','-'))) for i in l3]
[kernel3R.append(complex(i.replace('+-','-'))) for i in l3R]
[kernel5.append(complex(i.replace('+-','-'))) for i in l5]
[kernel5R.append(complex(i.replace('+-','-'))) for i in l5R]
[kernel7.append(complex(i.replace('+-','-'))) for i in l7]
[kernel7R.append(complex(i.replace('+-','-'))) for i in l7R]
[kernel9.append(complex(i.replace('+-','-'))) for i in l9]
[kernel9R.append(complex(i.replace('+-','-'))) for i in l9R]
[kernel11.append(complex(i.replace('+-','-'))) for i in l11]
[kernel11R.append(complex(i.replace('+-','-'))) for i in l11R]
[kernel13.append(complex(i.replace('+-','-'))) for i in l13]
[kernel13R.append(complex(i.replace('+-','-'))) for i in l13R]

kernel=np.array(kernel1)+np.array(kernel1R)+np.array(kernel3)+np.array(kernel3R)+np.array(kernel5)+np.array(kernel5R)+np.array(kernel7)+np.array(kernel7R)+np.array(kernel9)+np.array(kernel9R)+np.array(kernel11)+np.array(kernel11R)+np.array(kernel13)+np.array(kernel13R)
#kernel2=np.array(kernel1)+np.array(kernel1R)+np.array(kernel3)+np.array(kernel3R)+np.array(kernel5)+np.array(kernel5R)
yb=np.arange(-20,20,1e-1)
pbb=abs(np.array(kernel))**2
#pbb2=abs(np.array(kernel2))**2
plt.title("Probability for two step")
plt.xlabel("Screen coords(meters)")
plt.ylabel("Probability")
plt.plot(yb,pbb)
#plt.show()
plt.savefig("2step_todo.png")