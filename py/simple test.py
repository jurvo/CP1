# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 14:39:14 2021

@author: jurek
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')


### >>> BEGIN SETUP <<<
# mass
m_1, m_2 = 1, 1
# length of the pendulums
l_1, l_2 = 1, 1
# gravity const
g = 9.81

# inital angles
t_1_0, t_2_0 = np.pi/4., np.pi/2.
# inital momentum
v_1_0, v_2_0 = 0, 0

delta_t = 1/30
t_max = 10

### >>> END SETUP <<<

t = np.arange(0, t_max, delta_t)
t_1, t_2, v_1, v_2 = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))

t_1[0] = t_1_0
t_2[0] = t_2_0
v_1[0] = v_1_0
v_2[0] = v_2_0

delta_v_1, delta_v_2 = 1, 1

"""
delta_v_1 = - m_2/(m_1+m_2) * l_2/l_1 * (delta_v_2 * np.cos(t_1_0 - t_2_0) + v_2_0**2*np.sin(t_1_0 - t_2_0)) - g/l_1 * np.sin(t_1_0)

delta_v_2 = - l_1/l_2 * (delta_v_1 * np.cos(t_1_0 - t_2_0) + v_1_0**2*np.sin(t_1_0 - t_2_0)) - g/l_1 * np.sin(t_1_0)
"""

c_1 = - m_2/(m_1+m_2) * l_2/l_1
c_2 = - g/l_1
c_3 = - l_1/l_2
c_4 = - g/l_2

for i in range(1, len(t)):
    dt = t_1[i-1] - t_2[i-1]
    delta_v_1 = c_1 * (delta_v_2 * np.cos(dt) + v_2[i-1]**2*np.sin(dt)) + c_2 * np.sin(t_1[i-1])
    delta_v_2 = c_3 * (delta_v_1 * np.cos(dt) + v_1[i-1]**2*np.sin(dt)) + c_4 * np.sin(t_2[i-1])
    v_1[i] = v_1[i-1] + delta_t * delta_v_1
    t_1[i] = t_1[i-1] + delta_t * v_1[i]
    v_2[i] = v_2[i-1] + delta_t * delta_v_2
    t_2[i] = t_2[i-1] + delta_t * v_2[i]

#print(t_1)
#print(t_2)



fig = plt.figure()
ax = plt.axes(xlim=(-(l_1+l_2), l_1+l_2), ylim=(-(l_1+l_2), l_1+l_2))
line, = ax.plot([], [], lw=3)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    x = [0, l_1*np.sin(t_1[i]),l_1*np.sin(t_1[i])+l_2*np.sin(t_2[i])]
    y = [0, -l_1*np.cos(t_1[i]),-l_1*np.cos(t_1[i])-l_2*np.cos(t_2[i])]
    line.set_data(x, y)
    return line,



anim = FuncAnimation(fig, animate, init_func=init,
                               frames=int(t_max/delta_t), interval=int(delta_t*1000), blit=True)


#anim.save('sine_wave.gif', writer='imagemagick')
#anim.save('Pendulum_swing.mp4', fps=int(1/delta_t))
plt.show()