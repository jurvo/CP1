import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import functions

plt.style.use('seaborn-pastel')

### >>> BEGIN SETUP <<<

# EXACT 2 FILES ARE REQUIRED
files = ["default.txt", "default_1.txt"]
m = len(files)
fps = 30
verbose = True
animate_pendulum = True

if m != 2:
	print("Please specify two files!")
	quit()

### >>> END SETUP <<<
m1, m2, l1, l2, g, t_max, dt, sim_mode = [], [], [], [], [], [], [], []
t, q1, q2, p1, p2 = [], [], [], [], []
x1, x2, y1, y2 = [], [], [], []
E_kin, E_pot, E_tot = [], [], []
# load data and assign them
for i in range(m):
	if verbose: print("Load file ", str(i + 1), "of", str(m))
	settings, data = functions.loadDataFromFile(files[i])
	m1.append(settings["m1"])
	m2.append(settings["m2"])
	l1.append(settings["l1"])
	l2.append(settings["l2"])
	g.append(settings["g"])
	t_max.append(settings["tmax"])
	dt.append(settings["dt"])
	sim_mode.append(settings["sim"])

	t.append(data["t"])
	q1.append(data["q1"])
	q2.append(data["q2"])
	p1.append(data["p1"])
	p2.append(data["p2"])

	# calculating the x, y, e_kin, e_pot, e_tot values
	x1.append(l1[i]*np.sin(q1[i]))
	x2.append(x1[i] + l2[i]*np.sin(q2[i]))
	y1.append(-l1[i]*np.cos(q1[i]))
	y2.append(y1[i] - l2[i]*np.cos(q2[i]))

	E_kin.append(.5*m1[i]*l1[i]**2*p1[i]**2 + .5*m2[i]*(l1[i]**2*p1[i]**2 + l2[i]**2*p2[i]**2 + 2*p1[i]*p2[i]*l1[i]*l2[i]*np.cos(q1[i] - q2[i])))
	E_pot.append(-g[i]*(m1[i]*l1[i]*np.cos(q1[i]) + m2[i]*(l1[i]*np.cos(q1[i]) + l2[i]*np.cos(q2[i]))))
	E_tot.append(E_kin[i] + E_pot[i])

n = len(t[0])

if verbose:
	print("Loading complete.")
	print("Start animation.")

# plot the pendulum
if animate_pendulum:
	fig = plt.figure(figsize=(8,8))
	l = []
	for i in range(m):
		l.append(l1[i] + l2[i])
	L = max(l)
	ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, L))
	ax.set_aspect('equal')
	
	line1, = ax.plot([], [], 'o-', lw=2, color = "red")
	line2, = ax.plot([], [], 'o-', lw=2, color = "blue")

	time_template = 'time = %.2fs'
	time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


	def init():
		line1.set_data([], [])
		line2.set_data([], [])
		time_text.set_text(time_template % (0))
		return line1, line2, time_text

	def animate(i):
		line1.set_data([0, x1[0][i], x2[0][i]], [0, y1[0][i], y2[0][i]])
		line2.set_data([0, x1[1][i], x2[1][i]], [0, y1[1][i], y2[1][i]])
		time_text.set_text(time_template % (t[0][i]))
		return line1, line2, time_text
	
	anim = FuncAnimation(fig, animate, init_func=init, frames=np.arange(0, n, max(int(1/fps/dt[0]),1)), interval=1000/fps, blit = True)
	#if verbose: print("Start video rendering.")
	#anim.save("compare.mp4", fps=fps)
	#if verbose: print("Video rendering finished.")
	plt.show()