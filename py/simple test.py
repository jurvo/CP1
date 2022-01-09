import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import scipy
plt.style.use('seaborn-pastel')
# TODO: Compare with analytical solution for simple pendulum
### >>> BEGIN SETUP <<<
# mass
m_1, m_2 = 1, 1.
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

animate_pendulum = True
plot_energies = True
verbose = True

# 0 = Forward Euler, 1 = RK4
simulation_mode = 1

fps = 30
### >>> END SETUP <<<

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

# delcare and initilize arrays
t = np.arange(0, t_max, delta_t)
n = len(t)
t_1, t_2, v_1, v_2, a_1, a_2 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n) # create data arrays

# assing initial values
t_1[0], t_2[0] = t_1_0, t_2_0
v_1[0], v_2[0] = v_1_0, v_2_0
a_1[0], a_2[0] = 0., 0.

"Initial condition for Leapfrog algorithm"
# needs spatial coordinate info at time 1/2
phi_1 = phi_1_0 + 0.5*dt/(m*l*(1+(np.sin(phi_1_0-phi_2_0))**2))*(p_1_0-2*p_2_0*np.cos(phi_1_0-phi_2_0))
p_1=p_1_0-dt/(2*m*l)*((2*p_1_0*p_2_0*np.sin(phi_1_0-phi_2_0))*(1+(np.sin(phi_1_0-phi_2_0))**2)-2*np.sin(phi_1_0-phi_2_0)*np.cos(phi_1_0-phi_2_0)*(p_1_0**2+p_2_0**2-2*p_1_0*p_2_0*np.cos(phi_1_0-phi_2_0)))/((1+(np.sin(phi_1_0-phi_2_0))**2)**2)+2*m*g*l*np.sin(phi_1_0)
phi_2 = phi_2_0 + 0.5*dt/(m*l*(1+(np.sin(phi_1_0-phi_2_0))**2))*(p_2_0-2*p_1_0*np.cos(phi_1_0-phi_2_0))
p_2=p_2_0-dt/(2*m*l)*((-2*p_1_0*p_2_0*np.sin(phi_1_0-phi_2_0))*(1+(np.sin(phi_1_0-phi_2_0))**2)+2*np.sin(phi_1_0-phi_2_0)*np.cos(phi_1_0-phi_2_0)*(p_1_0**2+p_2_0**2-2*p_1_0*p_2_0*np.cos(phi_1_0-phi_2_0)))/((1+(np.sin(phi_1_0-phi_2_0))**2)**2)+m*g*l*np.sin(phi_2_0)

# a RK4 method
def phi(u, h ,f):
    k_1 = f(u)
    k_2 = f(u + .5 * h * k_1)
    k_3 = f(u + .5 * h * k_2)
    k_4 = f(u + h * k_3)
    return (k_1 + 2. * (k_2 + k_3) + k_4) / 6.

def G(u):
    t1 = u[0]
    t2 = u[1]
    v1 = u[2]
    v2 = u[3]
    dt = t1-t2
    alpha = c_1 * np.cos(dt)
    beta = c_1 * v2**2*np.sin(dt) + c_2*np.sin(t1)
    gamma = c_3 * np.cos(dt)
    delta = - c_3 * v1**2*np.sin(dt) + c_4*np.sin(t2)
    a1 = (alpha*delta + beta) / (1 - alpha * gamma)
    a2 = a1 * gamma + delta
    return np.array([v1, v2, a1, a2])

# do the integration
if verbose: print("Start time integration")
if simulation_mode == 0: # Forward Euler
    for i in range(1, n):
        dtheta = t_1[i-1] - t_2[i-1]
        a_1[i] = c_1 * (a_2[i-1] * np.cos(dtheta) + v_2[i-1]*v_2[i-1]*np.sin(dtheta)) + c_2 * np.sin(t_1[i-1])
        a_2[i] = c_3 * (a_1[i-1] * np.cos(dtheta) - v_1[i-1]*v_1[i-1]*np.sin(dtheta)) + c_4 * np.sin(t_2[i-1])
        v_1[i] = v_1[i-1] + delta_t * a_1[i]
        v_2[i] = v_2[i-1] + delta_t * a_2[i]
        t_1[i] = t_1[i-1] + delta_t * v_1[i]
        t_2[i] = t_2[i-1] + delta_t * v_2[i]
elif simulation_mode == 1: # RK4
    for i in range(1, n):
        p = delta_t * phi(np.array([t_1[i-1], t_2[i-1], v_1[i-1], v_2[i-1]]), delta_t, G)
        t_1[i] = t_1[i-1] + p[0]
        t_2[i] = t_2[i-1] + p[1]
        v_1[i] = v_1[i-1] + p[2]
        v_2[i] = v_2[i-1] + p[3]
if verbose:
    print("Time integration done.")
    print("Calculating...")
# calculating the x, y, e_kin, e_pot, e_tot values
x_1 = l_1*np.sin(t_1)
x_2 = x_1 + l_2*np.sin(t_2)
y_1 = -l_1*np.cos(t_1)
y_2 = y_1 - l_2*np.cos(t_2)


hist_x_1 = []
hist_y_1 = []
hist_phi_1 = []
hist_p_1 = []
hist_x_2 = []
hist_y_2 = []
hist_phi_2 = []
hist_p_2 = []

for iters in range(n_step):     

    p_1_new = p_1 + dt/(m*l*(1+(np.sin(phi_1-phi_2))**2))*(p_1-2*p_2*np.cos(phi_1-phi_2))
    p_2_new = p_2 + dt/(m*l*(1+(np.sin(phi_1-phi_2))**2))*(p_2-2*p_1*np.cos(phi_1-phi_2))
    phi_1_new = phi_1 + dt/(2*m*l)*((2*p_1_new*p_2_new*np.sin(phi_1-phi_2))*(1+(np.sin(phi_1-phi_2))**2)-2*np.sin(phi_1-phi_2)*np.cos(phi_1-phi_2)*(p_1_new**2+p_2_new**2-2*p_1_new*p_2_new*np.cos(phi_1-phi_2)))/((1+(np.sin(phi_1-phi_2))**2)**2)+2*m*g*l*np.sin(phi_1)
    phi_2_new = phi_2 + dt/(2*m*l)*((-2*p_1_new*p_2_new*np.sin(phi_1-phi_2))*(1+(np.sin(phi_1-phi_2))**2)+2*np.sin(phi_1-phi_2)*np.cos(phi_1-phi_2)*(p_1_new**2+p_2_new**2-2*p_1_new*p_2_new*np.cos(phi_1-phi_2)))/((1+(np.sin(phi_1-phi_2))**2)**2)+m*g*l*np.sin(phi_2)
  

    "Update the phi & p values"   
    p_v_1.append(p_1_new)
    phi_v_1.append(phi_1_new)
    p_v_2.append(p_2_new)
    phi_v_2.append(phi_2_new)
    
    phi_1 = phi_1_new
    p_1 = p_1_new
    phi_2 = phi_2_new
    p_2 = p_2_new
    
    "Record and extend t "
    t = np.append(t, (iters+1)*dt)
    

    "Limit phi to range of [-4, 4]"
    if (phi_1 > 4): phi_1 = phi_1 - 2.0*np.pi
    if (phi_1 < -4): phi_1 = phi_1 + 2.0*np.pi
    if (phi_2 > 4): phi_2 = phi_2 - 2.0*np.pi
    if (phi_2 < -4): phi_2 = phi_2 + 2.0*np.pi

    x_1, y_1, x_2, y_2 = beam(phi_1, phi_2)

#    hist_x_1.append(x_1)
#    hist_y_1.append(y_1)
#    hist_phi_1.append(phi_1)
#    hist_p_1.append(p_1)
#    hist_x_2.append(x_2)
#    hist_y_2.append(y_2)
#    hist_phi_2.append(phi_2)
#    hist_p_2.append(p_2)


## the spatial coordinate calculated in leapfrog is advanced of
# hald a timestep w.r.t. momentum
## this is the ``real" time at which our phi are evaluated
t_phi= t+ 1/2.*dt
## now interpolate to t
phi_v_1= np.interp(t, t_phi, phi_v_1)
phi_v_2= np.interp(t, t_phi, phi_v_2)
# setting phi_0, since this cannot be calculated form interpolation
phi_v_1[0]= phi_1_0
phi_v_2[0]= phi_2_0
    
    
"calculate energy"
#E= hamiltonian(phi_v_1, p_v_1, phi_v_2, p_v_2)


#print('Method: Leapfrog DE=' , max(E)- min(E))

"Plotting related codes: "
"Phi & p phase space contour"
phi_contour, p_contour = np.meshgrid(np.linspace(-1.5*np.pi, 1.5*np.pi, 100), \
                                      np.linspace(-2.5, 2.5, 50))  

'''h = hamiltonian(phi_contour, p_contour)

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

point, = ax.plot([phi_0], [p_0], 'or')
line, = ax1.plot([], [], 'o-', lw=2)
E_graph, = ax2.plot([], [])

"function to set compute data into animation frames"
def animate(i):
    t_list = t[:i]
    E_list = E[:i]
    thisx = [0, hist_x[i]]
    thisy = [0, hist_y[i]]

    point.set_data(hist_phi[i],hist_p[i])
    line.set_data(thisx, thisy)
    E_graph.set_data(t_list, E_list)
    return point, line, E_graph,


ani = animation.FuncAnimation(
    fig, animate, len(hist_y), interval=dt*1000, blit=True)

# ani.save('Pendulum_swing.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()'''

E_kin = .5*l**2*p_1**2 + .5*(l**2*p_1**2 + l**2*p_2**2 + 2*p_1*p_2*l*l*np.cos(phi_1 - phi_2))
E_pot = -g*(m*l*np.cos(phi_1) + m*(l*np.cos(phi_1) + l*np.cos(phi_2)))
E_tot = E_kin + E_pot
if verbose:
    print("Calculations done.")
    print("Start plotting.")
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
        x = [0, x_1[i], x_2[i]]
        y = [0, y_1[i], y_2[i]]
        line.set_data(x, y)
        time_text.set_text(time_template % (t[i]))
        return line,time_text
    
    anim = FuncAnimation(fig, animate, init_func=init,frames=np.arange(0, n, max(int(1/fps/delta_t),1)), interval=1000/fps, blit = True)
    #anim.save('sine_wave.gif', writer='imagemagick')
    #anim.save("Pendulum_swing.mp4", fps=int(1/delta_t))
    plt.show()

# plot the energies

fig = plt.figure()
plt.plot(t, E_kin, label="Kinetic")
plt.plot(t, E_pot, label="Potentail")
plt.plot(t, E_tot, label="Total")

    plt.ylim([-50,200])
    plt.legend()
    plt.show()
if verbose: print("Plotting done.")
