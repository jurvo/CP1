import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import functions

plt.style.use('seaborn-pastel')

### >>> BEGIN SETUP <<<

file = "RK4_dt.1_tmax10000000.txt"
fps = 30
verbose = True
animate_pendulum = False
plot_energies = True

### >>> END SETUP <<<

# load data and assign them
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

if verbose:
    print("Calculations done.")
    print("Start plotting.")
# plot the pendulum
if animate_pendulum:
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(xlim=(-(l1+l2), l1+l2), ylim=(-(l1+l2), l1+l2))
    line, = ax.plot([], [], 'o-',lw=2)
    time_template = 'time = %.2fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = [0, x1[i], x2[i]]
        y = [0, y1[i], y2[i]]
        line.set_data(x, y)
        time_text.set_text(time_template % (t[i]))
        return line,time_text
    
    anim = FuncAnimation(fig, animate, init_func=init,frames=np.arange(0, n, max(int(1/fps/dt),1)), interval=1000/fps, blit = True)
    #anim.save('sine_wave.gif', writer='imagemagick')
    #anim.save("Pendulum_swing.mp4", fps=int(1/delta_t))
    plt.show()

# plot the energies
if plot_energies:
    fig = plt.figure()

    plt.plot(t, E_kin, label="Kinetic")
    plt.plot(t, E_pot, label="Potential")
    plt.plot(t, E_tot, label="Total")

    plt.ylim([-50,200])
    plt.legend()
    plt.title(file)
    #plt.text(0, 175, 'delta_t=')
    #plt.text(15, 175, dt)
    #plt.text(0, 160, 't_max=')
    #plt.text(15, 160, t_max)
    print("Saving...")
    plt.savefig('RK4_dt.1_tmax10000000.png')
    print("Saving done!")
    plt.show()
if verbose: print("Plotting done.")