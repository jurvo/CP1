import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

def hamiltonian(phi, p):
    return 0.5*p**2 + 1. - np.cos(phi)

def beam(phi):
    x = np.sin(phi)
    y = -np.cos(phi)
    return x, y

def RK_pendulum(t, state):
    phi, p = state
    return np.array([p, -np.sin(phi)])

solve_ivp_opt = 'Basic' # 'Basic', 'Step_control'

"First Task"
# Set the option to 'Basic' and run the code. Compare the energy
# conservation behaviour with your own implemented RK4 method. Does it perform better
# or worse? 
    
# Read up on the scipy RK4 implementation:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
# Focus on the 'RK45' method description.
# can you give a short explanation for the observed behaviour?

"Second Task"
# Set the option to 'Step_control' and run the code. Compare the energy conservation 
# behaviour once again. Does the energy conservation behaviour improve? 

# Can you explain what effect it has, when you use the combination of 
# (max_step, atol, rtol) arguments?

"Third Task"
# Can you give a conclusion, whether RK4 method is symplectic or not?

"Timesteps"
dt = 0.1
t_span = (0.0, 20.)
n_step = int(t_span[-1]/dt)
t_val = np.linspace(t_span[0], t_span[-1], n_step)

"Initial Conditions"
state_0 = [0.0, 1.2]

"Scipy implementation of RK4"
if solve_ivp_opt == 'Basic':
    hist_state = solve_ivp(RK_pendulum, t_span, state_0, method='RK45', t_eval = t_val)
elif solve_ivp_opt == 'Step_control':
    hist_state = solve_ivp(RK_pendulum, t_span, state_0, method='RK45', t_eval = t_val, max_step=0.01, atol = 1, rtol = 1)

t = t_val
E = hamiltonian(hist_state.y[0], hist_state.y[1])
hist_x, hist_y = beam(hist_state.y[0])



"Phi & p phase space contour"
phi_contour, p_contour = np.meshgrid(np.linspace(-1.5*np.pi, 1.5*np.pi, 100), \
                                      np.linspace(-2.5, 2.5, 50))  

h = hamiltonian(phi_contour, p_contour)

"Plotting related functions"
fig = plt.figure(constrained_layout=False)
fig.set_size_inches(9, 6)#(18.5, 9.0)
gs = fig.add_gridspec(ncols=2, nrows=2)

"preparing subplot for phase space contour plot"
ax = fig.add_subplot(gs[0, 0])
ax.contourf(phi_contour, p_contour, h)
ax.set_aspect('equal')
ax.set_xlabel('phi')
ax.set_ylabel('p')

"preparing subplot for pendulum position plot"
ax1 = fig.add_subplot(gs[0, 1])
ax1.set_ylim(-1.5, 1.5)
ax1.set_xlim(-1.5, 1.5)
ax1.set_aspect('equal')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

"preparing subplot for energy over time plot"
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_xlim(min(t), max(t))
ax2.set_ylim(min(E), max(E))
ax2.grid()
ax2.set_xlabel('t')
ax2.set_ylabel('E')

point, = ax.plot([0.0], [1.2], 'or')
line, = ax1.plot([], [], 'o-', lw=2)
E_graph, = ax2.plot([], [])

"function to set compute data into animation frames"
def animate(i):
    t_list = t[:i]
    E_list = E[:i]
    thisx = [0, hist_x[i]]
    thisy = [0, hist_y[i]]

    point.set_data(hist_state.y[0][i],hist_state.y[1][i])
    line.set_data(thisx, thisy)
    E_graph.set_data(t_list, E_list)
    return point, line, E_graph,


ani = animation.FuncAnimation(
    fig, animate, len(hist_y), interval=dt*1000, blit=True)

# ani.save('test_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()
