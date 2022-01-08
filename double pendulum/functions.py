import numpy as np

def saveDataToFile(filename, time, angle_1, angle_2, momentum_1, momentum_2, m_1, m_2, l_1, l_2, g, t_max, delta_t, sim_mode):
	file = open(filename, "w")
	file.write("#	t	q1	q2	p1	p2\n")
	file.write("# t	Time\n")
	file.write("# q1/q2	generalized coordinates\n")
	file.write("# p1/p2	generalized momentum\n")
	file.write("# Simulation settings:\n")
	file.write("# Mass 1: " + str(m_1) + "; Mass 2: " + str(m_2) + "\n")
	file.write("# Length 1: " + str(l_1) + "; Length 2: " + str(l_2) + "\n")
	file.write("# Gravity: " + str(g) + "\n")
	file.write("# max Time: " + str(t_max) + "\n")
	file.write("# Timestep: " + str(delta_t) + "\n")
	s = "Forward Euler" if sim_mode == 0 else "RK4"
	file.write("#	Simulation method: " + s + "\n")

	for i in range(len(time)):
		file.write(str(time[i]) + "	" + str(angle_1[i]) + "	" + str(angle_2[i]) + "	" + str(momentum_1[i]) + "	" + str(momentum_2[i]) + "\n")
	file.close()

def loadDataFromFile(filename):
	data = np.genfromtxt (filename, comments ='#', delimiter=None, skip_header=0, usecols=None, names=True)
	return data["t"], data["q1"], data["q2"], data["p1"], data["p2"]