import numpy as np

DATAPATH = ".\\data\\"
TAB = "\t"

def saveDataToFile(filename, time, angle_1, angle_2, momentum_1, momentum_2, m_1, m_2, l_1, l_2, g, t_max, delta_t, sim_mode):
	file = open(DATAPATH + filename, "w")
	file.write("#	m1	m2	l1	l2	g	tmax	dt	sim\n")
	file.write(str(m_1) + TAB + str(m_2) + TAB + str(l_1) + TAB + str(l_2) + TAB + str(g) + TAB + str(t_max) + TAB + str(delta_t) + TAB + str(sim_mode) + "\n")
	file.write("#	t	q1	q2	p1	p2\n")
	file.write("# t	Time\n")
	file.write("# q1/q2	generalized coordinates\n")
	file.write("# p1/p2	generalized momentum\n")
	for i in range(len(time)):
		file.write(str(time[i]) + "	" + str(angle_1[i]) + "	" + str(angle_2[i]) + "	" + str(momentum_1[i]) + "	" + str(momentum_2[i]) + "\n")
	file.close()

def loadDataFromFile(filename):
	settings = np.genfromtxt (DATAPATH + filename, comments ='#', delimiter=None, skip_header=0, usecols=None, max_rows=1, names=True)
	data = np.genfromtxt (DATAPATH + filename, comments ='#', delimiter=None, skip_header=2, usecols=None, names=True)
	return settings, data