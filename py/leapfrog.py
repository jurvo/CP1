import numpy as np


'Hamiltonian'

def kinetic(q1, q2, p1, p2):
    return 1/(2*m*l) * (p1**2+p2**2-2*p1*p2*np.cos(q1-q2))/(1+(np.sin(q1-q2))**2)

def potential(q1,q2):
    return m*g*l*(3-2*np.cos(q1)-np.cos(q2))

def Etot(q1, q2, p1, p2):
    return kinetic(q1, q2, p1, p2) + potential(q1, q2)

'Initial Conditions'

q10=np.pi/2     #angle of mass 1
q20=0           #angle of mass 2 in respect of mass 1
p10=0           #momentum of mass 1
p20=0           #momentum of mass 2
m=1             #mass of 1 and 2
l=1             #length
g=9.81          #gravity constant


'Time Initals'

dt=0.01         #time steps
tf=1            #final time
n=int(tf/dt)    #number of steps


'Initialize arrays'

q1a=np.zeros(n)
q2a=np.zeros(n)
p1a=np.zeros(n)
p2a=np.zeros(n)


'Initial conditions for leapfrog algorithm (for q at time 1/2)'

q1 = q10 + 0.5*dt/(m*l**2*(1+(np.sin(q10-q20))**2))*(p10-p20*np.cos(q10-q20))
q2 = q20 + 0.5*dt/(m*l**2*(1+(np.sin(q10-q20))**2))*(2*p20-p10*np.cos(q10-q20))
p1=p10
p2=p20


'Assigning initial conditions for leapfrog to arrays'

q1a[0]=q1
q2a[0]=q2
p1a[0]=p1
p2a[0]=p2


'loop for leapfrog algorithm'

for i in range(n):
    q1new = q1 + dt/(m*l**2*(1+(np.sin(q1-q2))**2))*(p1-p2*np.cos(q1-q2))
    q2new = q2 + dt/(m*l**2*(1+(np.sin(q1-q2))**2))*(2*p2-p1*np.cos(q1-q2))
    p1new = p1 - dt/(2*m*l**2)*((2*p1*p2*np.sin(q1new-q2new))*(1+(np.sin(q1new-q2new))**2)-2*np.sin(q1new-q2new)*np.cos(q1new-q2new)*(p1**2+p2**2-2*p1*p2*np.cos(q1new-q2new)))/((1+(np.sin(q1new-q2new))**2)**2) + 2*m*g*l*np.sin(q1new)
    p2new = p2 + dt/(2*m*l**2)*((2*p1*p2*np.sin(q1new-q2new))*(1+(np.sin(q1new-q2new))**2)-2*np.sin(q1new-q2new)*np.cos(q1new-q2new)*(p1**2+p2**2-2*p1*p2*np.cos(q1new-q2new)))/((1+(np.sin(q1new-q2new))**2)**2) + m*g*l*np.sin(q2new)


    'Updating arrays'
    q1a[i]=q1new
    q2a[i]=q2new
    p1a[i]=p1new
    p2a[i]=p2new


    'Updating q and p for next step in loop'
    
    q1=q1new
    q2=q2new
    p1=p1new
    p2=p2new

    print('E_kin:', kinetic(q1, q2, p1, p2),'E_pot:', potential(q1, q2), 'E_tot:',  Etot(q1, q2, p1, p2))