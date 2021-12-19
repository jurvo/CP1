# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 14:39:14 2021

@author: jurek
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


### >>> BEGIN SETUP <<<
# mass
m_1, m_2 = 1, 1
# length of the pendulums
l_1, l_2 = 1, 1
# gravity const
g = 9.81

# inital angles
t_1, t_2 = np.pi/4., np.pi/4.
# inital momentum
v_1, v_2 = 0, 0

delta_t = .1
t_max = 10

### >>> END SETUP <<<

