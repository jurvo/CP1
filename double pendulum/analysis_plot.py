import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import functions

plt.style.use('seaborn-pastel')

filesFE = ["FE0.001.txt","FE0.005.txt","FE0.010.txt","FE0.050.txt","FE0.100.txt","FE0.500.txt", "FE1.000.txt","ref.txt"]
filesRK = ["RK0.001.txt","RK0.005.txt","RK0.010.txt","RK0.050.txt","RK0.100.txt","RK0.500.txt","RK1.000.txt", "ref.txt"]
filesFEF = ["FEF0.001.txt","FEF0.005.txt","FEF0.010.txt","FEF0.050.txt","FEF0.100.txt","FEF0.500.txt", "FEF1.000.txt","ref.txt"]

files = filesFEF

ta = []
E_tota = []
E_kina = []
E_pota = []
# load data and assign them
for file in files:
	settings, data = functions.loadDataFromFile(file)
	m1, m2, l1, l2, g, t_max, dt, sim_mode = settings["m1"], settings["m2"], settings["l1"], settings["l2"], settings["g"], settings["tmax"], settings["dt"], settings["sim"]
	t, q1, q2, p1, p2 = data["t"], data["q1"], data["q2"], data["p1"], data["p2"]
	n = len(t)

	# calculating the x, y, e_kin, e_pot, e_tot values
	x1 = l1*np.sin(q1)
	x2 = x1 + l2*np.sin(q2)
	y1 = -l1*np.cos(q1)
	y2 = y1 - l2*np.cos(q2)
	E_kin = .5*m1*l1**2*p1**2 + .5*m2*(l1**2*p1**2 + l2**2*p2**2 + 2*p1*p2*l1*l2*np.cos(q1 - q2))
	E_pot = -g*(m1*l1*np.cos(q1) + m2*(l1*np.cos(q1) + l2*np.cos(q2)))
	E_tot = E_kin + E_pot

	ta.append(t)
	E_tota.append(E_tot)
	E_kina.append(E_kin)
	E_pota.append(E_pota)
	
fig = plt.figure()
for i in range(len(E_tota)):
#	plt.plot(t, E_kin, label="Kinetic")
#	plt.plot(t, E_pot, label="Potentail")
	plt.plot(ta[i], E_tota[i], label=files[i])

plt.ylim([-50,200])
plt.legend()
plt.show()