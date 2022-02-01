import numpy as np
from functions import *
import threading
import time

def phi(u, h ,f, c_1, c_2, c_3, c_4):
	k_1 = f(u, c_1, c_2, c_3, c_4)
	k_2 = f(u + .5 * h * k_1, c_1, c_2, c_3, c_4)
	k_3 = f(u + .5 * h * k_2, c_1, c_2, c_3, c_4)
	k_4 = f(u + h * k_3, c_1, c_2, c_3, c_4)
	return (k_1 + 2. * (k_2 + k_3) + k_4) / 6.

def G(u, c_1, c_2, c_3, c_4):
	t1 = u[0]
	t2 = u[1]
	v1 = u[2]
	v2 = u[3]
	dt = t1-t2
	alpha = c_1 * np.cos(dt)
	beta = c_1 * v2**2*np.sin(dt) + c_2*np.sin(t1)
	gamma = c_3 * np.cos(dt)
	delta = - c_3 * v1**2*np.sin(dt) + c_4*np.sin(t2)
	a1 = (alpha*delta + beta) / (1 - alpha * gamma)
	a2 = a1 * gamma + delta
	return np.array([v1, v2, a1, a2])

def simulate(m_1, m_2, l_1, l_2, g, t_1_0, t_2_0, v_1_0, v_2_0, t_max, delta_t, simulation_mode, verbose, filename):
	if verbose:
		print("Double Pendulum simulation:")
		print("m1:", m_1, "| m2:", m_2)
		print("l1:", l_1, "| l2:", l_2)
		print("g:", g)
		print("Initial Values:")
		print("t1:", t_1_0, "| t2:", t_2_0)
		print("v1:", v_1_0, "| v2:", v_2_0)
		print("Time settings: from 0 to", t_max, "seconds in ", delta_t, "sec steps")
		print("Simulation is done with:", "Forward Euler" if simulation_mode == 0 else "RK4")

	# delcare and initilize arrays
	t = np.arange(0, t_max, delta_t)
	n = len(t)
	t_1, t_2, v_1, v_2, a_1, a_2 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

	# assing initial values
	t_1[0], t_2[0] = t_1_0, t_2_0
	v_1[0], v_2[0] = v_1_0, v_2_0
	a_1[0], a_2[0] = 0., 0.

	# calculating the constants
	c_1 = - m_2/(m_1+m_2) * l_2/l_1
	c_2 = - g/l_1
	c_3 = - l_1/l_2
	c_4 = - g/l_2

	# do the integration
	if verbose: print("Start time integration")
	if simulation_mode == 0: # Forward Euler
		for i in range(1, n):
			dtheta = t_1[i-1] - t_2[i-1]
			a_1[i] = c_1 * (a_2[i-1] * np.cos(dtheta) + v_2[i-1]*v_2[i-1]*np.sin(dtheta)) + c_2 * np.sin(t_1[i-1])
			a_2[i] = c_3 * (a_1[i-1] * np.cos(dtheta) - v_1[i-1]*v_1[i-1]*np.sin(dtheta)) + c_4 * np.sin(t_2[i-1])
			v_1[i] = v_1[i-1] + delta_t * a_1[i]
			v_2[i] = v_2[i-1] + delta_t * a_2[i]
			t_1[i] = t_1[i-1] + delta_t * v_1[i]
			t_2[i] = t_2[i-1] + delta_t * v_2[i]
	elif simulation_mode == 1: # RK4
		for i in range(1, n):
			p = delta_t * phi(np.array([t_1[i-1], t_2[i-1], v_1[i-1], v_2[i-1]]), delta_t, G, c_1, c_2, c_3, c_4)
			t_1[i] = t_1[i-1] + p[0]
			t_2[i] = t_2[i-1] + p[1]
			v_1[i] = v_1[i-1] + p[2]
			v_2[i] = v_2[i-1] + p[3]

	if verbose:
		print("Time integration done.")
		print("Saving...")
	saveDataToFile(filename, t, t_1, t_2, v_1, v_2, m_1, m_2, l_1, l_2, g, t_max, delta_t, simulation_mode)
	if verbose: print("Saving done!")

### >>> BEGIN SETUP <<<
# masses
m_1, m_2 = 1., 1.
# length of the pendulums
l_1, l_2 = 1., 1.
# gravity const
g = 9.81

# inital angles
t_1_0, t_2_0 = np.pi/2,np.pi/2
#t_1_0, t_2_0 = 2*np.pi/3., -np.pi/18.
# inital momentum
v_1_0, v_2_0 = 0.0, 0.0

# in seconds
delta_t = 0.001
t_max = 100

verbose = False

# 0 = Forward Euler, 1 = RK4
simulation_mode = 1
#filename = "default1.txt"
### >>> END SETUP <<<

simulate(m_1, m_2, l_1, l_2, g, t_1_0, t_2_0, v_1_0, v_2_0, t_max, delta_t, simulation_mode, verbose, "default.txt")

"""
def thread_func(x, y):
	print(x, y)
	simulate(m_1, m_2, l_1, l_2, g, x, y, v_1_0, v_2_0, t_max, delta_t, simulation_mode, verbose, "fractal\\" + str(x) + ";" + str(y) +".txt")
	print("done")
	return

def work(foo):
    foo.do_task()

from multiprocessing import Pool

class t:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def do_task(self):
		print(self.x, self.y)
		simulate(m_1, m_2, l_1, l_2, g, self.x, self.y, v_1_0, v_2_0, t_max, delta_t, simulation_mode, verbose, "fractal\\" + str(self.x) + ";" + str(self.y) +".txt")
		print("done")

init = np.linspace(start = -np.pi, stop =  np.pi, num = 50, endpoint = False)

tasks = []
for x in init:
	for y in init:
		tasks.append(t(x,y))

if __name__ == '__main__':

	pool = Pool()
	pool.map(work, tasks)
	pool.close()
	pool.join()

"""