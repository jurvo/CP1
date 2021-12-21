import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import scipy
plt.style.use('seaborn-pastel')



### >>> BEGIN SETUP <<<
# mass
m_1, m_2 = 1., 1.
# length of the pendulums
l_1, l_2 = 1., 1.
# gravity const
g = 9.81

# inital angles
t_1_0, t_2_0 = 2*np.pi/3., -np.pi/18.
# inital momentum
v_1_0, v_2_0 = 0.0, 0.0

delta_t = .02
t_max = 5
n = 1. # take only each n-th datapoint for the animation

animate_pendulum = True
plot_energies = True

### >>> END SETUP <<<

# delcare and initilize arrays
t = np.arange(0, t_max, delta_t)
t_1, t_2, v_1, v_2, a_1, a_2 = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)) # create data arrays

# assinge initial values
t_1[0], t_2[0] = t_1_0, t_2_0
v_1[0], v_2[0] = v_1_0, v_2_0
a_1[0], a_2[0] = 0., 0.

# calculating the constants
c_1 = - m_2/(m_1+m_2) * l_2/l_1
c_2 = - g/l_1
c_3 = - l_1/l_2
c_4 = - g/l_2

# do the integration

for i in range(1, len(t)):
    dtheta = t_1[i-1] - t_2[i-1]
    a_1[i] = c_1 * (a_2[i-1] * np.cos(dtheta) + v_2[i-1]**2*np.sin(dtheta)) + c_2 * np.sin(t_1[i-1])
    a_2[i] = c_3 * (a_1[i-1] * np.cos(dtheta) + v_1[i-1]**2*np.sin(dtheta)) + c_4 * np.sin(t_2[i-1])
    v_1[i] = v_1[i-1] + delta_t * a_1[i]
    v_2[i] = v_2[i-1] + delta_t * a_2[i]
    t_1[i] = t_1[i-1] + delta_t * v_1[i]
    t_2[i] = t_2[i-1] + delta_t * v_2[i]

# calculating the x, y, e_kin, e_pot, e_tot values
x_1 = l_1*np.sin(t_1)
x_2 = x_1 + l_2*np.sin(t_2)
y_1 = -l_1*np.cos(t_1)
y_2 = y_1 - l_2*np.cos(t_2)

E_kin = .5*m_1*l_1**2*v_1**2 + .5*m_2*(l_1**2*v_1**2 + l_2**2*v_2**2 + 2*v_1*v_2*l_1*l_2*np.cos(t_1 - t_2))
E_pot = -g*(m_1*l_1*np.cos(t_1) + m_2*(l_1*np.cos(t_1) + l_2*np.cos(t_2)))
E_tot = E_kin + E_pot

# plot the pendulum
if animate_pendulum:

    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(xlim=(-(l_1+l_2), l_1+l_2), ylim=(-(l_1+l_2), l_1+l_2))
    line, = ax.plot([], [], 'o-',lw=2)
    time_template = 'time = %.2fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        i = int(n * i)
        x = [0, x_1[i], x_2[i]]
        y = [0, y_1[i], y_2[i]]
        line.set_data(x, y)
        time_text.set_text(time_template % (i*delta_t))
        return line,time_text
    
    anim = FuncAnimation(fig, animate, init_func=init,
                                frames=int(t_max/delta_t), interval=int(delta_t*1000/n), blit=True)
    #anim.save('sine_wave.gif', writer='imagemagick')
    #anim.save("Pendulum_swing.mp4", fps=int(1/delta_t))
    plt.show()

# plot the energies
if plot_energies:
    fig = plt.figure()

    plt.plot(t, E_kin, label="Kinetic")
    plt.plot(t, E_pot, label="Potentail")
    plt.plot(t, E_tot, label="Total")

    plt.ylim([-50,200])
    plt.legend()
    plt.show()