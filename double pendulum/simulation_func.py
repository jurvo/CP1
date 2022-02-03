import numpy as np
from functions import *
import threading
import time

# function for RK4, u[i+1] = u[i] + h * phi()
def phi(u, h ,f, args):
	k_1 = f(u, args)
	k_2 = f(u + .5 * h * k_1, args)
	k_3 = f(u + .5 * h * k_2, args)
	k_4 = f(u + h * k_3, args)
	return h * (k_1 + 2. * (k_2 + k_3) + k_4) / 6.

# function for RK4, df/dt = G
def G(u, args):
	# init
	q1, q2 = u[0], u[1]
	p1, p2 = u[2], u[3]
	dq = q1-q2
	c_1, c_2, c_3, c_4 = args[0], args[1], args[2], args[3]
	alpha, beta = c_1 * np.cos(dq), c_1 * p2**2*np.sin(dq) + c_2*np.sin(q1)
	gamma, delta = c_3 * np.cos(dq), - c_3 * p1**2*np.sin(dq) + c_4*np.sin(q2)
	# calculating G1 and G2
	G1 = (alpha*delta + beta) / (1 - alpha * gamma)
	G2 = G1 * gamma + delta
	return np.array([p1, p2, G1, G2])

def simulateDoublePendulum(m_1, m_2, l_1, l_2, g, t_1_0, t_2_0, v_1_0, v_2_0, t_max, delta_t, simulation_mode, verbose, filename):
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
		print("Initializing.")

	# delcare and initilize arrays
	t = np.arange(0, t_max, delta_t)
	n = len(t)
	p = np.zeros((6, n))

	# assing initial values
	p[0, 0] = t_1_0
	p[1, 0] = t_2_0
	p[2, 0] = v_1_0
	p[3, 0] = v_2_0
	p[4, 0], p[5, 0] = 0., 0.

	# calculating the constants
	c_1 = - m_2/(m_1+m_2) * l_2/l_1
	c_2 = - g/l_1
	c_3 = - l_1/l_2
	c_4 = - g/l_2

	args = np.array([c_1, c_2, c_3, c_4])

	t_stop = n
	modulus = 1 / delta_t
	# do the integration
	if verbose: print("Initializing finished.\nStarting time integration.")
	if simulation_mode == 0: # Forward Euler
		for i in range(1, n):
			# p[:,i] = [q1[i], q2[i], p1[i], p2[i], a1[i], a2[i]]
			dq = p[0, i-1] - p[1, i-1]
			# a_1 = c_1 * (a_2 * cos(dt) + v_2**2 * sin(dt)) + c_2 * sin(t_1)
			p[4, i] = c_1 * (p[5, i-1] * np.cos(dq) + p[3, i-1]*p[3, i-1]*np.sin(dq)) + c_2 * np.sin(p[0,i-1])
			# a_2 = c_3 * (a_1 * cos(dt) + v_1**2 * sin(dt)) + c_4 * sin(t_2)
			p[5, i] = c_3 * (p[4, i-1] * np.cos(dq) - p[2, i-1]*p[2, i-1]*np.sin(dq)) + c_4 * np.sin(p[1,i-1])
			# p1[i] = p1[i-1] + dt * a_1
			p[2, i] = p[2, i-1] + delta_t * p[4, i]
			# p2[i] = p2[i-1] + dt * a_2
			p[3, i] = p[3, i-1] + delta_t * p[5, i]
			# q1[i] = q1[i-1] + dt * p1[i]
			p[0, i] = p[0, i-1] + delta_t * p[2, i]
			# q2[i] = q2[i-1] + dt * p2[i]
			p[1, i] = p[1, i-1] + delta_t * p[3, i]
			if verbose and i % modulus == 0: print(i, "of", n, "done.")
			if not np.isfinite(p[:,i]).all():
				if verbose: print("Break due to overflow at index", i, "of", n)
				t_stop = i - 1
				break
	elif simulation_mode == 1: # RK4
		for i in range(1, n):
			# p[i] = p[i-1] + dt * phi()
			p_prev = p[0:4, i-1]
			p[0:4, i] = p_prev + phi(p_prev, delta_t, G, args)
			if verbose and i % modulus == 0: print(i, "of", n, "done.")
			if not np.isfinite(p[:,i]).all():
				if verbose: print("Break due to overflow at index", i, "of", n)
				t_stop = i - 1
				break
	elif simulation_mode == 2: # fixed FE
		for i in range(1, n):
			# p[i] = p[i-1] + dt * G()
			p_prev = p[0:4, i-1]
			p[0:4, i] = p_prev + delta_t * G(p_prev, args)
			if verbose and i % modulus == 0: print(i, "of", n, "done.")
			if not np.isfinite(p[:,i]).all():
				if verbose: print("Break due to overflow at index", i, "of", n)
				t_stop = i - 1
				break

	if verbose: print("Time integration done.\nSaving into", filename)
	saveDataToFile(filename, t[:t_stop], p[0, :t_stop], p[1, :t_stop], p[2, :t_stop], p[3, :t_stop], m_1, m_2, l_1, l_2, g, t_max, delta_t, simulation_mode)
	if verbose: print("Saving done!")

"""
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
filename = "default.txt"
### >>> END SETUP <<<

simulateDoublePendulum(m_1, m_2, l_1, l_2, g, t_1_0, t_2_0, v_1_0, v_2_0, t_max, delta_t, simulation_mode, verbose, filename)
"""