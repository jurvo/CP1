import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool

plt.style.use('seaborn-pastel')
# TODO: Compare with analytical solution for simple pendulum
### >>> BEGIN SETUP <<<
# mass
m_1, m_2 = 1., 01.
# length of the pendulums
l_1, l_2 = 1., 1.
# gravity const
g = 9.81

# inital angles
t_1_0, t_2_0 =  np.pi/4 , - np.pi/4
#t_1_0, t_2_0 = np.pi/4, 0

# initial value array values
initmin = -(np.pi -0.01) # -0.01 to avoid stable state in upright position 
initmax = (np.pi -0.01)
initres = 10 # resolution of measured points, for square array only
t_max = 10
delta_t = 0.001
border_of_exacticity = 2 # threshold value, to avoid time consuming computation of "impossible" values. Set to 2 for acceptable results, little bit higher saves time but destroys exacticity.

# inital momentum
v_1_0, v_2_0 = 0.0, 0.0

# in seconds


animate_pendulum = False
plot_energies = False
verbose = True
plot_phasespace = False
plot_phasespacekorr = False
flipcounter = False
calculate_fractal = True #RK4 only, switches to RK4 automatically.

# 0 = Forward Euler, 1 = RK4
simulation_mode = 1

################## >>> END SETUP <<<
#constants, to be needed later
c_1 = - m_2/(m_1+m_2) * l_2/l_1
c_2 = - g/l_1
c_3 = - l_1/l_2
c_4 = - g/l_2
# initial value arrays
# here obviously t_1_init could be used, but, if other shapes of arrays should be needed it's easier to build the code directly for two dims instead of a double use of one dim.

if verbose:
    print("Double Pendulum simulation:")
    print("m1:", m_1, "| m2:", m_2)
    print("l1:", l_1, "| l2:", l_2)
    print("g:", g)
    print("Initial Values:")
#    print("initmin:", initmin, "| initmax:", initmax, "|resolution:", initres)
    print("v1:", v_1_0, "| v2:", v_2_0)
#    print("Time settings: from 0 to", t_max, "seconds in ", delta_t, "sec steps")
    print("Simulation is done with:", "Forward Euler" if simulation_mode == 0 else "RK4")

# delcare and initilize arrays
t = np.arange(0, t_max, delta_t)
t_1, t_2, v_1, v_2, a_1, a_2 = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)) # create data arrays

# functions:
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

def fliptime(t_max, t1in, t2in, m, l, verbose, valueprint): # begin of the Function, returns breakvalue only, given Values are to be: t_1_init[m] , t_2_init[l] , (v_1_0, v_2_0 maybe those are global ones?) , border_of_exacticity , t_max , t
    #assigning initial values
    breakvalue = t_max #breakvalue inside the function, t_max given from outside
    t_1[0], t_2[0] = t1in, t2in # Values given to the Function
    v_1[0], v_2[0] = v_1_0, v_2_0 # Globs (Global values)
    a_1[0], a_2[0] = 0., 0. # Zeros (zero Values)
    
    # Filter for "impossible" values, where energy is not enough for Flip, to save calculation time
    if (3*np.cos(t_1[0])+np.cos(t_2[0])) > border_of_exacticity: #border... is Glob
        breakvalue = t_max 
    else:     
        #begin of the RK4 function
        # a RK4 method,  
        ####funktionen außerhalb loop setzen!
        

        # do the integration
        if verbose: print("Start time integration for t_1_init =", m, " and t_2_init =", l) 
        # RK4
        for i in range(1, len(t)): # integrate until it ends, then set breakvalue to the right value. better, the other way 'round
            breakvalue = i * delta_t # breakvalue (defined outside the loop, so can be used outside, too) is set to actual number of iteration, 
            p = delta_t * phi(np.array([t_1[i-1], t_2[i-1], v_1[i-1], v_2[i-1]]), delta_t, G)
            t_1[i] = t_1[i-1] + p[0]
            t_2[i] = t_2[i-1] + p[1]
            v_1[i] = v_1[i-1] + p[2]
            v_2[i] = v_2[i-1] + p[3]
            if t_1[i]>np.pi  or t_1[i]<-np.pi or t_2[i]>np.pi or t_2[i]<-np.pi : #flip check break, breaks if Flip is detected, breakvalue will then be timepoint of Flip.
                break
        if valueprint: print("m=", m, "=", t_1[0], "| l=", l, "=", t_2[0], "breakvalue =", breakvalue )
    return breakvalue

def calculate_fractal(t_max, initmin, initmax, verbose, valueprint):
    t_1_init = np.linspace(initmin, initmax, initres, endpoint=True)
    t_2_init = np.linspace(initmin, initmax, initres, endpoint=True)
    fractal = np.zeros((initres, initres)) # array definitions outside the loops, so they can be used after the loop is done   
    for l in range(len(t_2_init)): # begin to loop over t_2-Values
        # fractal_row = [] #initialisation of all rows of the fractal map, means looping over t_2 values, one value per row.
        # breakvalue = t_max # it is better to leave this on max, to avoid errors if calculation breaks
        for m in range(len(t_1_init)): #initialisation of the row's value loop, loop over t_1 values with const t_2
            breakvalue = fliptime(t_max, t_1_init[m], t_2_init[l], m, l, verbose, valueprint)
            fractal[l, m] = breakvalue # fractal_row.append(breakvalue) # appending breakvalue to row 
    return fractal
#def logization(fractal):#generating Log of Fractal values
#    logfractal = [] 
#    for i in range(1, len(fractal)): 
#        logfractal_row = []
#        for j in range(0, len(fractal[i])):
#            log = np.log10(fractal[i][j])
#            logfractal_row.append(log)
#        logfractal.append(logfractal_row)
#    return logfractal

# calculating the constants, to be used later




#pool = Pool()
#quadrant1 = pool.apply_async(calculate_fractal, [t_max, 0, (np.pi -0.01), False, False])
#quadrant2 = pool.apply_async(calculate_fractal, [t_max, (np.pi -0.01), 0 , False, False])
#quadrant3 = pool.apply_async(calculate_fractal, [t_max, -(np.pi -0.01), (np.pi -0.01), False, False])
#quadrant4 = pool.apply_async(calculate_fractal, [t_max, 0, -(np.pi -0.01), False, False])


#Plot as colour-coded field of squares
fractal= calculate_fractal(t_max, initmin, initmax, True, False)
#fractal= calculate_fractal(10, -(np.pi -0.01), 0, True, False)
logfractal = np.log(fractal)
np.save('fractal_data_InitRes='+str(initres)+'_time='+str(t_max)+'_timeStep='+str(delta_t), fractal)
#fractal = calculate_fractal(t_max, initmin, initmax, True, False)
fig = plt.figure()
plt.pcolormesh(logfractal)
plt.xlabel('fractal_data_InitRes:_'+str(initres)+'_time:_'+str(t_max)+'_timeStep:_'+str(delta_t))
#plt.tight_layout()
plt.show()

    

    
    



















