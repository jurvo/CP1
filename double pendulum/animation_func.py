import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import functions

plt.style.use('seaborn-pastel')

def animateDoublePendulum(file, fps, animate = True, verbose = False, save = False):
	# load data and assign them
	if verbose: print("Load data from file", file)
	settings, data = functions.loadDataFromFile(file)
	m1, m2, l1, l2, g, t_max, dt, sim_mode = settings["m1"], settings["m2"], settings["l1"], settings["l2"], settings["g"], settings["tmax"], settings["dt"], settings["sim"]
	t, q1, q2, p1, p2 = data["t"], data["q1"], data["q2"], data["p1"], data["p2"]
	n = len(t)
	if verbose: print("Loading complete.\nStarting calculation.")
	# calculating the x, y, e_kin, e_pot, e_tot values
	x1 = l1*np.sin(q1)
	x2 = x1 + l2*np.sin(q2)
	y1 = -l1*np.cos(q1)
	y2 = y1 - l2*np.cos(q2)
	E_kin = .5*m1*l1**2*p1**2 + .5*m2*(l1**2*p1**2 + l2**2*p2**2 + 2*p1*p2*l1*l2*np.cos(q1 - q2))
	E_pot = -g*(m1*l1*np.cos(q1) + m2*(l1*np.cos(q1) + l2*np.cos(q2)))
	E_tot = E_kin + E_pot
	if verbose: print("Calculation complete.\nStart plotting.")
	if animate:
		# plot the pendulum
		fig = plt.figure(figsize=(8,8))
		gs = fig.add_gridspec(ncols=2, nrows=2)
		time_template = 'time = %.2fs'

		# prepare realspace plot
		ax = fig.add_subplot(gs[0,0])
		ax.set_xlim(-(l1+l2), l1+l2)
		ax.set_ylim(-(l1+l2), l1+l2)
		ax.set_aspect('equal')
		ax.set_xlabel("X")
		ax.set_ylabel("Y")
		time_text = ax.text(0.05, 0.9, '0 s', transform=ax.transAxes)

		# prepare Energy-time plot
		ax1 = fig.add_subplot(gs[1,0])
		ax1.set_xlim(min(t), max(t))
		ax1.set_ylim(min(min(E_tot), min(E_kin), min(E_pot)), max(max(E_tot), max(E_kin), max(E_pot)))
		ax1.grid()
		ax1.set_xlabel("Time / s")
		ax1.set_ylabel("Energy / arb. units")

		# prepare phasespace plot 1
		ax2 = fig.add_subplot(gs[0,1])
		ax2.set_xlim(min(q1), max(q1))
		ax2.set_ylim(min(p1), max(p1))
		ax2.set_xlabel(r"$\vartheta_1$")
		ax2.set_ylabel(r"$p_1$")

		# prepare phasespace plot 2
		ax3 = fig.add_subplot(gs[1,1])
		ax3.set_xlim(min(q2), max(q2))
		ax3.set_ylim(min(p2), max(p2))
		#ax3.set_xlim(min(min(q1), min(q2)), max(max(q1), max(q2)))
		#ax3.set_ylim(min(min(p1), min(p2)), max(max(p1), max(p2)))
		ax3.set_xlabel(r"$\vartheta_2$")
		ax3.set_ylabel(r"$p_2$")

		# init plots
		## realspace
		line, = ax.plot([0, x1[0], x2[0]], [0, y1[0], y2[0]], 'o-',lw=2)
		## energy time
		E_tot_graph, = ax1.plot([],[]) 
		E_kin_graph, = ax1.plot([],[])
		E_pot_graph, = ax1.plot([],[])
		## phase space
		point1, = ax2.plot([q1[0]], [p1[0]], '')
		point2, = ax3.plot([q2[0]], [p2[0]], '')

		def animate(i):
			x = [0, x1[i], x2[i]]
			y = [0, y1[i], y2[i]]
			t_list = t[:i]
			E_tot_list = E_tot[:i]
			E_kin_list = E_kin[:i]
			E_pot_list = E_pot[:i]
			
			q1_list = q1[:i]
			q2_list = q2[:i]
			p1_list = p1[:i]
			p2_list = p2[:i]
			
			line.set_data(x, y)
			E_tot_graph.set_data(t_list, E_tot_list)
			E_kin_graph.set_data(t_list, E_kin_list)
			E_pot_graph.set_data(t_list, E_pot_list)
			point1.set_data(q1_list, p1_list)
			point2.set_data(q2_list, p2_list)
			time_text.set_text(time_template % (t[i]))
			return line, E_tot_graph, E_kin_graph, E_pot_graph, point1, point2, time_text
		
		anim = FuncAnimation(fig, animate, frames=np.arange(0, n, max(int(1/fps/dt),1)), interval=1000/fps, blit = True)
		if save:
			if verbose: print("Video saving...")
			anim.save(file + ".mp4", fps=fps)
			if verbose: print("Video saving done.")
		else:
			plt.show()
	else:
		# plot the energies
		fig = plt.figure()

		plt.plot(t, E_kin, label="Kinetic")
		plt.plot(t, E_pot, label="Potentail")
		plt.plot(t, E_tot, label="Total")

		plt.ylim([-50,200])
		plt.legend()
		plt.show()
"""
### >>> BEGIN SETUP <<<
file = "default.txt"
fps = 30
animate_pendulum = True
plot_energies = False
### >>> END SETUP <<<

animateDoublePendulum(file, fps)
"""