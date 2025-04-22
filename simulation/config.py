import numpy as np

WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (0, 0, 0)
PARTICLE_COLOR = (0, 0, 255)

DELTA_T = 0.01  # Time step for simulation
GRAVITY = np.array([0, 9.81 * DELTA_T], dtype='float64')  # Gravity vector (x, y)
DAMPING_FACTOR = 1.0
LOW_INITAL_VECOLITY = -5
HIGH_INITIAL_VELOCITY = 5
LOW_ELASTICITY = 0.5  
HIGH_ELASTICITY = 0.9  