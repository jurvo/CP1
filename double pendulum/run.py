from matplotlib import animation
import numpy as np
import simulation_func as sf
import animation_func as af
import datetime

### >>> BEGIN SETUP <<<
# mass
m_1, m_2 = 1., 1.
# length of the pendulums
l_1, l_2 = 1., 1.
# gravity const
g = 9.81

# inital angles
#t_1_0, t_2_0 = np.pi/2,np.pi/2
t_1_0, t_2_0 = np.pi/3., -np.pi/18.
# inital momentum
v_1_0, v_2_0 = 0.0, 0.0

# time setup in seconds
delta_t = 0.001
t_max = 100

# verbose mode
verbose = True

# 0 = Forward Euler, 1 = RK4, 2 = fixed FE
simulation_mode = 1
filename = "ref.txt"
animate = True
saveToMovie = False
### >>> END SETUP <<<
now = datetime.datetime.now()
sf.simulateDoublePendulum(m_1, m_2, l_1, l_2, g, t_1_0, t_2_0, v_1_0, v_2_0, t_max, delta_t, simulation_mode, verbose, filename)
print(datetime.datetime.now() - now)
af.animateDoublePendulum(filename, 30, animate, verbose, saveToMovie)