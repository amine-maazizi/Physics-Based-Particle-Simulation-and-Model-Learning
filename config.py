import numpy as np

WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (0, 0, 0)
PARTICLE_COLOR = (0, 0, 255)

GRAVITY = np.array([0, 0.1]) 
DAMPING_FACTOR = 0.99 
LOW_INITAL_VECOLITY = -2
HIGH_INITIAL_VELOCITY = 2
LOW_ELASTICITY = 0.5  
HIGH_ELASTICITY = 1.0  