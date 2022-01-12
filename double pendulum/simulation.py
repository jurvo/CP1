import numpy as np
from functions import *

### >>> BEGIN SETUP <<<

simulation_mode = 1     # 0 = Forward Euler, 1 = RK4
comparison_mode = True     # False = one pendulum, True = two pendulum comparison
verbose = True

# masses
m = [1, 1, 1, 1]
# length of the pendulums
l = [1, 1, 1, 1]
# gravity const
g = 9.81

# inital angles
q_0 = [np.pi/2, np.pi/2, np.pi/2, np.pi/2]
# inital momentum
v_0 = [0, 0, 0, 0]
# in seconds
delta_t = 0.001
t_max = 100

### >>> END SETUP <<<

if verbose:
    if comparison_mode:
        print("Double Pendulum simulation in comparison:")
        print()
        print("Settings for first setup:")
        print("m1:", m[0], "| m2:", m[1])
        print("l1:", l[0], "| l2:", l[1])
        print("g:", g)
        print("Initial Values:")
        print("t1:", q_0[0], "| t2:", q_0[1])
        print("v1:", v_0[0], "| v2:", v_0[1])
        print()
        print("Settings for second setup:")
        print("m1:", m[2], "| m2:", m[3])
        print("l1:", l[2], "| l2:", l[3])
        print("g:", g)
        print("Initial Values:")
        print("t1:", q_0[2], "| t2:", q_0[3])
        print("v1:", v_0[2], "| v2:", v_0[3])
        print()
        print("Time settings: from 0 to", t_max, "seconds in ", delta_t, "sec steps")
        print("Simulation is done with:", "Forward Euler" if simulation_mode == 0 else "RK4")
        
    else:
        print("Double Pendulum simulation:")
        print("m1:", m[0], "| m2:", m[1])
        print("l1:", l[0], "| l2:", l[1])
        print("g:", g)
        print("Initial Values:")
        print("t1:", q_0[0], "| t2:", q_0[1])
        print("v1:", v_0[0], "| v2:", v_0[1])
        print("Time settings: from 0 to", t_max, "seconds in ", delta_t, "sec steps")
        print("Simulation is done with:", "Forward Euler" if simulation_mode == 0 else "RK4")       

# delcare and initilize arrays
t = np.arange(0, t_max, delta_t)
n = len(t)
q=[np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n)]
v=[np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n)]
a=[np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n)]


# assing initial values
q[0][0], q[1][0] = q_0[0], q_0[1]
v[0][0], v[1][0] = v_0[0], v_0[1]
a[0][0], a[1][0] = 0., 0.

#calculating the constants
c=[np.zeros(4),np.zeros(4)]
c[0][0] = - m[1]/(m[0]+m[1]) * l[1]/l[0]
c[0][1] = - g/l[0]
c[0][2] = - l[0]/l[1]
c[0][3] = - g/l[1]

if comparison_mode:
    q[2][0], q[3][0] = q_0[2], q_0[3]
    v[2][0], v[3][0] = v_0[2], v_0[3]
    a[2][0], a[3][0] = 0., 0.
    
    c[1][0] = - m[3]/(m[2]+m[3]) * l[3]/l[2]
    c[1][1] = - g/l[2]
    c[1][2] = - l[2]/l[3]
    c[1][3] = - g/l[3]


# a RK4 method
def phi(u, h ,f):
	k_1 = f(u)
	k_2 = f(u + .5 * h * k_1)
	k_3 = f(u + .5 * h * k_2)
	k_4 = f(u + h * k_3)
	return (k_1 + 2. * (k_2 + k_3) + k_4) / 6.

def G(u):
	t1 = u[0]
	t2 = u[1]
	v1 = u[2]
	v2 = u[3]
	dt = t1-t2
	alpha = c[0][0] * np.cos(dt)
	beta = c[0][0] * v2**2*np.sin(dt) + c[0][1]*np.sin(t1)
	gamma = c[0][2] * np.cos(dt)
	delta = - c[0][2] * v1**2*np.sin(dt) + c[0][3]*np.sin(t2)
	a1 = (alpha*delta + beta) / (1 - alpha * gamma)
	a2 = a1 * gamma + delta
	return np.array([v1, v2, a1, a2])


# do the integration
if verbose: print("Start time integration")
if simulation_mode == 0: # Forward Euler
	for i in range(1, n):
		dq = q[0][i-1] - q[1][i-1]
		a[0][i] = c[0][0] * (a[1][i-1] * np.cos(dq) + v[1][i-1]*v[1][i-1]*np.sin(dq)) + c[0][1] * np.sin(q[0][i-1])
		a[1][i] = c[0][2] * (a[0][i-1] * np.cos(dq) - v[0][i-1]*v[0][i-1]*np.sin(dq)) + c[0][3] * np.sin(q[1][i-1])
		v[0][i] = v[0][i-1] + delta_t * a[0][i]
		v[1][i] = v[1][i-1] + delta_t * a[1][i]
		q[0][i] = q[0][i-1] + delta_t * v[0][i]
		q[1][i] = q[1][i-1] + delta_t * v[1][i]
elif simulation_mode == 1: # RK4
	for i in range(1, n):
		p0 = delta_t * phi(np.array([q[0][i-1], q[1][i-1], v[0][i-1], v[1][i-1]]), delta_t, G)
		q[0][i] = q[0][i-1] + p0[0]
		q[1][i] = q[1][i-1] + p0[1]
		v[0][i] = v[0][i-1] + p0[2]
		v[1][i] = v[1][i-1] + p0[3]

if comparison_mode:
    if simulation_mode == 0: # Forward Euler
    	for i in range(1, n):
		    dq = q[2][i-1] - q[3][i-1]
		    a[2][i] = c[1][0] * (a[3][i-1] * np.cos(dq) + v[3][i-1]*v[3][i-1]*np.sin(dq)) + c[1][1] * np.sin(q[2][i-1])
		    a[3][i] = c[1][2] * (a[2][i-1] * np.cos(dq) - v[2][i-1]*v[2][i-1]*np.sin(dq)) + c[1][3] * np.sin(q[3][i-1])
		    v[2][i] = v[2][i-1] + delta_t * a[2][i]
		    v[3][i] = v[3][i-1] + delta_t * a[3][i]
		    q[2][i] = q[2][i-1] + delta_t * v[2][i]
		    q[3][i] = q[3][i-1] + delta_t * v[3][i]
    elif simulation_mode == 1: # RK4
	    for i in range(1, n):
		    p1 = delta_t * phi(np.array([q[0][i-1], q[1][i-1], v[0][i-1], v[1][i-1]]), delta_t, G)
		    q[2][i] = q[2][i-1] + p1[0]
		    q[3][i] = q[3][i-1] + p1[1]
		    v[2][i] = v[2][i-1] + p1[2]
		    v[3][i] = v[3][i-1] + p1[3]


if verbose:
	print("Time integration done.")
	print("Saving...")
saveDataToFile("default0.txt", t, q[0], q[1], v[0], v[1], m[0], m[1], l[0], l[1], g, t_max, delta_t, simulation_mode)
if comparison_mode: saveDataToFile("default1.txt", t, q[2], q[3], v[2], v[3], m[2], m[3], l[2], l[3], g, t_max, delta_t, simulation_mode)
if verbose: print("Saving done!")