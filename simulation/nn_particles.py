import pygame
import numpy as np
import sys
import torch

from simulation.config import *
from simulation.utils import elasticity_to_color

sys.path.append('..')
from models.model_architectures import ParticleNet, ParticleNetSCFC

# Load the trained model
model_name = 'particle_net_scfc' 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ParticleNet().to(device)
model.load_state_dict(torch.load(f'models/{model_name}.pth', map_location=device))
model.eval()
print(f'Loaded {model_name}.pth for inference')

class NNParticle:
    def __init__(self, x, y, vx, vy, elasticity=0.5, radius=5):
        self.initial_velocity = np.array([vx, vy], dtype='float64')
        self.position = np.array([x, y], dtype='float64')
        self.radius = radius
        self.elasticity = elasticity
        self.color = elasticity_to_color(self.elasticity, channel=1)
        # Simulation parameters for neural network input
        self.g = GRAVITY[1]  # Gravity magnitude (e.g., 9.81 m/s²)
        self.dt = DELTA_T    # Time step (e.g., 0.01 s)
        self.x_min = 0       # Left boundary (e.g., 0)
        self.x_max = WIDTH   # Right boundary (e.g., WIDTH)
        self.step = 0        # Step number for tracking time (n)

    def update(self):
        # Prepare input vector for neural network
        input_vec = np.array([
            self.position[0],          # x
            HEIGHT - self.position[1], # y (inverted for screen space)
            self.initial_velocity[0],  # v_x0
            -self.initial_velocity[1], # v_y0 (inverted for screen space)
            -self.g,                   # g (negative due to screen space)
            self.dt,                   # Δt
            self.elasticity,           # e
            self.x_min,                # x_min
            self.x_max,                # x_max
            self.step                  # n (step number)
        ], dtype='float32')

        # Convert to PyTorch tensor and move to device
        input_tensor = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)

        # Get predicted next position [x(t+Δt), y(t+Δt)]
        with torch.no_grad():
            predicted_state = model(input_tensor).squeeze(0).cpu().numpy()

        # Update position in pygame coordinates
        self.position = np.array([predicted_state[0], HEIGHT - predicted_state[1]], dtype='float64')  # [x, HEIGHT - y]

        # Increment step number
        self.step += 1
    
    def draw(self, surface):
        draw_position = (np.float64(self.position[0]), np.float64(self.position[1] - self.radius))
        pygame.draw.circle(surface, self.color, draw_position, self.radius)