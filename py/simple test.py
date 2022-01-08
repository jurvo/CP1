import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import RK45


'Hamiltonian of the pendulum system'
def hamiltonian(phi, p):
    return 1/(2*m*l**2)*(p_1**2+p_2**2-2*p_1*p_2*np.cos(phi_1-phi_2)/(1+(np.sin(phi_1-phi_2))**2) +m*g*l*(3-2*np.cos(phi_1)-np.cos(phi_2)))

'X & Y coordinate of the pendulum'
def beam(phi):
    x_1 = l*np.sin(phi_1)
    y_1 = -l*np.cos(phi_1)
    x_2 = l*np.sin(phi_1)+l*np.sin(phi_2)
    y_2 = -l*np.cos(phi_1)-l*np.cos(phi_2)

    return x_1, y_1, x_2, y_2


"Timesteps"
dt = 0.1 # 0.01, 0.1
t_end = 20
n_step = int(t_end/dt)

"Initial Conditions"
m=1
l=1
g=9.81
phi_1_0=0.0
p_1_0=1.2
phi_2_0=0
p_2_0=0

"Initial time and energy"
t = np.zeros(1)


"Initial condition for Leapfrog algorithm"
# needs spatial coordinate info at time 1/2
phi_1 = phi_1_0 + 0.5*dt/(m*l*(1+(np.sin(phi_1_0-phi_2_0))**2))*(p_1_0-2*p_2_0*np.cos(phi_1_0-phi_2_0))
p_1=p_1_0-dt/(2*m*l)*((2*p_1_0*p_2_0*np.sin(phi_1_0-phi_2_0))*(1+(np.sin(phi_1_0-phi_2_0))**2)-2*np.sin(phi_1_0-phi_2_0)*np.cos(phi_1_0-phi_2_0)*(p_1_0**2+p_2_0**2-2*p_1_0*p_2_0*np.cos(phi_1_0-phi_2_0)))/((1+(np.sin(phi_1_0-phi_2_0))**2)**2)+2*m*g*l*np.sin(phi_1_0)
phi_2 = phi_2_0 + 0.5*dt/(m*l*(1+(np.sin(phi_1_0-phi_2_0))**2))*(p_2_0-2*p_1_0*np.cos(phi_1_0-phi_2_0))
p_2=p_2_0-dt/(2*m*l)*((-2*p_1_0*p_2_0*np.sin(phi_1_0-phi_2_0))*(1+(np.sin(phi_1_0-phi_2_0))**2)+2*np.sin(phi_1_0-phi_2_0)*np.cos(phi_1_0-phi_2_0)*(p_1_0**2+p_2_0**2-2*p_1_0*p_2_0*np.cos(phi_1_0-phi_2_0)))/((1+(np.sin(phi_1_0-phi_2_0))**2)**2)+m*g*l*np.sin(phi_2_0)


phi_v_1= [phi_1]
p_v_1= [p_1]
phi_v_2= [phi_2]
p_v_2= [p_2]


hist_x_1 = []
hist_y_1 = []
hist_phi_1 = []
hist_p_1 = []
hist_x_2 = []
hist_y_2 = []
hist_phi_2 = []
hist_p_2 = []

for iters in range(n_step):     

    p_1_new = p_1 + (-np.sin(phi_1))*dt
    phi_1_new = phi_1 + dt*p_1_new
    p_2_new = p_2 + (-np.sin(phi_2))*dt
    phi_2_new = phi_2 + dt*p_2_new
  

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

    x_1, y_1 = beam(phi_1)
    x_2, y_2 = beam(phi_1)

    hist_x_1.append(x_1)
    hist_y_1.append(y_1)
    hist_phi_1.append(phi_1)
    hist_p_1.append(p_1)
    hist_x_2.append(x_2)
    hist_y_2.append(y_2)
    hist_phi_2.append(phi_2)
    hist_p_2.append(p_2)


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
E= hamiltonian(phi_v_1, p_v_1, phi_v_2, p_v_2)


print('Method: Leapfrog DE=' , max(E)- min(E))

"Plotting related codes: "
"Phi & p phase space contour"
phi_contour, p_contour = np.meshgrid(np.linspace(-1.5*np.pi, 1.5*np.pi, 100), \
                                      np.linspace(-2.5, 2.5, 50))  

h = hamiltonian(phi_contour, p_contour)

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
plt.show()