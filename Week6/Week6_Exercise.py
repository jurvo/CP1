import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import RK45

# 3 time animated subplots will be saved when the code is run, they show:
# subplot 1: phase space; the contours corresponds to possible trajectories
# the trajectory depends on the initial conditions
# subplot 2: pendulum evolution in xy space
# subplot 3: energy evolution as a function of time

# solve the equations with forward Euler and dt=0.01 and dt=0.1
# what do you notice? is this expected behavior?

# implement a leapfrog algorithm. use dt=0.1. What changes?

# is leapfrog "better" or "worse" than Forward Euler? Why?

# implement a Verlet algorithm. How do energy evolution compare with leapfrog? 

# what is the difference between leapfrog and Verlet in this case?

# Does Verlet algorithm conserve energy better than Leapfrog algorithm?

# Can you compare them in terms of computational cost?

# What do you have to keep into account for leapfrog/Verlet algorithm
# when calculating the energy?

# Now solve the equation numerically use the trapezoidal integration method:
# what is the energy evolution?

# The derived Runge-Kutta method is already implemented.
# Does RK4 conserve energy over time?


'Hamiltonian of the pendulum system'
def hamiltonian(phi, p):
    return 0.5*np.multiply(p,p) + 1. - np.cos(phi)

'X & Y coordinate of the pendulum'
def beam(phi):
    x = np.sin(phi)
    y = -np.cos(phi)
    return x, y

'functions to be used in Newton-Raphson method'
def f(dt, phi, p, new_value):
    return 2./dt*new_value + dt/2.*np.sin(new_value)-2./dt

def f_prime(dt, new_value):
    return

'Newton-Raphson method for solving pendulum position'
def Pendulum_eqn(dt, phi, p):
    delta = 1e-16
    new_value = 0.75
    iterations = 0
    for i in range(100):
        residual = 0.0 # What is the right expression for residual?
        
        if (abs(residual) < delta):
            iterations = i
            break
        
        "Newton - Raphson method"
    
    if (iterations == 0):
        print("Solution did not converge! Residue at: ", residual)
    return new_value, residual, iterations

"Timesteps"
dt = 0.01 # 0.01, 0.1
t_end = 20
n_step = int(t_end/dt)

"Initial Conditions"
phi_0=0
p_0=1.2


"Initial time and energy"
t = np.zeros(1)


"We can switch between methods here."
ODE_method = 'Verlet' # 'Forward', 'Leapfrog', 'Verlet', 'Integration', 'RK4'

if (ODE_method == 'Verlet'):
    "Initial condition for Verlet algorithm"
    # needs velocity info at time -1/2
    p = p_0 - dt/2. * np.sin(phi_0)
    phi= phi_0
    
    
elif (ODE_method == 'Leapfrog'):
    "Initial condition for Leapfrog algorithm"
    # needs spatial coordinate info at time 1/2
    phi = phi_0 + dt/2. * p_0
    p= p_0
    
else:
    phi=phi_0
    p=p_0
    
phi_v = [phi]
p_v = [p]


hist_x = []
hist_y = []
hist_phi = []
hist_p = []
for iters in range(n_step):
    if (ODE_method == 'Forward'):
        "Forward Euler"
        phi_new = phi + dt*p
        p_new = p - dt*np.sin(phi)
        
    elif (ODE_method == 'Leapfrog'):
        '''
        Leapfrog algorithm
        Since the phi value is already offset by half a step in the initial condition,
        the update to p_new and phi_new is ALMOST similar to Forward Euler.
        '''
        p_new = p - dt * np.sin(phi)
        phi_new = phi + dt* p_new
  
    elif (ODE_method == 'Verlet'):
        '''
        Verlet algorithm
        Since the p value is already offset by half a step in the initial condition,
        the update to p_new and phi_new is ALMOST similar to Forward Euler.
        '''
        p_new = p - dt * np.sin(phi)
        phi_new = phi + dt* p_new
    
    elif (ODE_method == 'Integration'):
        '''
        Integration method
        First, compute either phi_new or p_new using Newton-Raphson method.
        Highly recommend to solve for phi_new, since it is a much simpler expression.
        '''
        new_value, res, its = Pendulum_eqn(dt, phi, p)
        p_new = 0.0
        phi_new = 0.0

    elif (ODE_method == 'RK4'):
        '''
        Runge-Kutta 4th order method
        You must realise that phi is a function of p & t. phi = phi(t, p)
        Also understand that p is a function of phi & t. p = p(t, phi)
        Follow your derived steps to calculate phi_new!
        '''
        p1 = p
        phi1 = phi
        k1 = p1*dt
        
        p2 = p1 - np.sin(phi1 + k1*0.5)*dt*0.5
        k2 = p2*dt
        
        p3 = p1 - np.sin(phi1 + k2*0.5)*dt*0.5
        k3 = p3*dt
        
        p4 = p1 - np.sin(phi1 + k3)*dt
        k4 = p4*dt
        
        phi_new = phi + (k1 + 2.*k2 + 2.*k3 + k4)/6.
        
        'Next, calculate p_new'
        f1 = -np.sin(phi1)*dt
        phi2 = phi1 + (p1 + f1*0.5)*dt*0.5
        
        f2 = -np.sin(phi2)*dt
        phi3 = phi1 + (p1 + f2*0.5)*dt*0.5
        
        f3 = -np.sin(phi3)*dt
        phi4 = phi1 + dt*(p1 + f3)
        
        f4 = -np.sin(phi4)*dt
        
        p_new = p + (f1 + 2.*f2 + 2.*f3 + f4)/6.
    
    else:
        print('Invalid method chosen.')
    
    "Update the phi & p values"   
    p_v.append(p_new)
    phi_v.append(phi_new)
    
    phi = phi_new
    p = p_new
    
    "Record and extend t "
    t = np.append(t, (iters+1)*dt)
    

    "Limit phi to range of [-4, 4]"
    if (phi > 4): phi = phi - 2.0*np.pi
    if (phi < -4): phi = phi + 2.0*np.pi

    x, y = beam(phi)

    hist_x.append(x)
    hist_y.append(y)
    hist_phi.append(phi)
    hist_p.append(p)




if ODE_method== 'Leapfrog':
    ## the spatial coordinate calculated in leapfrog is advanced of
    # hald a timestep w.r.t. momentum
    ## this is the ``real" time at which our phi are evaluated
    t_phi= t+ 1/2.*dt
    ## now interpolate to t
    phi_v= np.interp(t, t_phi, phi_v)
    # setting phi_0, since this cannot be calculated form interpolation
    phi_v[0]= phi_0
    
elif ODE_method== 'Verlet':
    ## the momentum calculated in Verlet is retarded of
    # hald a timestep w.r.t. spatial coordinate
    
    # calculating the last value of the moentum
    p_last= p_v[-1]- np.sin(phi_v[-1])*dt/2
    
    ## this is the ``real" time at which our phi are evaluated
    t_p= t- 1/2.*dt
    ## now interpolate to t
    p_v= np.interp(t, t_p, p_v)
    # fixing the last point, that cannot be calculated by interpolation
    p_v[-1]= p_last
   
    
    
    
"calculate energy"
E= hamiltonian(phi_v, p_v)


if ODE_method=='Leapfrog' or ODE_method=='Verlet' or ODE_method=='Integration':
    print('Method: ', ODE_method, 'DE=' , max(E)- min(E))

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