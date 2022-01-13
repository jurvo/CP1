import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import functions

plt.style.use('seaborn-pastel')

### >>> BEGIN SETUP <<<

files = ["default.txt"]
m = len(files)
fps = 30
verbose = True
animate_pendulum = True

plot_energies = True

### >>> END SETUP <<<
m1 = m2 = l1 = l2 = g = t_max = dt = sim_mode = m*[None]
t = q1 = q2 = p1 = p2 = m*[None]
x1 = x2 = y1 = y2 = m*[None]
E_kin = E_pot = E_tot = m*[None]
# load data and assign them
for i in range(m):
	settings, data = functions.loadDataFromFile(files[i])
	m1[i], m2[i], l1[i], l2[i], g[i], t_max[i], dt[i], sim_mode[i] = settings["m1"], settings["m2"], settings["l1"], settings["l2"], settings["g"], settings["tmax"], settings["dt"], settings["sim"]
	t[i], q1[i], q2[i], p1[i], p2[i] = data["t"], data["q1"], data["q2"], data["p1"], data["p2"]
	
	# calculating the x, y, e_kin, e_pot, e_tot values
	x1[i] = l1[i]*np.sin(q1[i])
	x2[i] = x1[i] + l2[i]*np.sin(q2[i])
	y1[i] = -l1[i]*np.cos(q1[i])
	y2[i] = y1[i] - l2[i]*np.cos(q2[i])

	E_kin[i] = .5*m1[i]*l1[i]**2*p1[i]**2 + .5*m2[i]*(l1[i]**2*p1[i]**2 + l2[i]**2*p2[i]**2 + 2*p1[i]*p2[i]*l1[i]*l2[i]*np.cos(q1[i] - q2[i]))
	E_pot[i] = -g[i]*(m1[i]*l1[i]*np.cos(q1[i]) + m2[i]*(l1[i]*np.cos(q1[i]) + l2[i]*np.cos(q2[i])))
	E_tot[i] = E_kin[i] + E_pot[i]

n = len(t[0])

if verbose:
	print("Calculations done.")
	print("Start plotting.")

# plot the pendulum
if animate_pendulum:
	fig = plt.figure(figsize=(8,8))
	ax = plt.axes(xlim=(-(l1[0]+l2[0]), l1[0]+l2[0]), ylim=(-(l1[0]+l2[0]), l1[0]+l2[0]))
	line = m*[None]
	for	i in range(m):
		line[i], = ax.plot([], [], 'o-',lw=2)
	time_template = 'time = %.2fs'
	time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

	def init():
		for	i in range(m):
			line[i].set_data([], [])
		return line,

	def animate(i):
		for	j in range(m):
			x = [0, x1[j][i], x2[j][i]]
			y = [0, y1[j][i], y2[j][i]]
			line[j].set_data(x, y)
		time_text.set_text(time_template % (t[0][i]))
		return line,time_text
	
	anim = FuncAnimation(fig, animate, init_func=init,frames=np.arange(0, n, max(int(1/fps/dt[0]),1)), interval=1000/fps, blit = True)
	#anim.save('sine_wave.gif', writer='imagemagick')
	#anim.save("Pendulum_swing.mp4", fps=int(1/delta_t))
	# SUBPLOTS!!!
	plt.show()
"""
# plot the energies
if plot_energies:
	fig = plt.figure()

	plt.plot(t, E_kin, label="Kinetic")
	plt.plot(t, E_pot, label="Potentail")
	plt.plot(t, E_tot, label="Total")

	plt.ylim([-50,200])
	plt.legend()
	plt.show()
if verbose: print("Plotting done.")
"""