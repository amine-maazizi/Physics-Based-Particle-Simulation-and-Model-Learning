import pygame
import numpy as np
import random
import torch
import torch.nn as nn

from simulation.config import *
from simulation.utils import elasticity_to_color

# Neural Network Architecture (must match the trained model)
class ParticleNet(nn.Module):
    def __init__(self):
        super(ParticleNet, self).__init__()
        # First hidden layer: Predict free motion state [x', y', vx', vy']
        self.free_motion = nn.Sequential(
            nn.Linear(11, 6),  # Input: [x, y, vx, vy, e, vx0, vy0, g, Δt, x_min, x_max]
            nn.ReLU(),
            nn.Linear(6, 4)  # Output: [x', y', vx', vy']
        )
        # Second hidden layer: Predict collision flags [c_ground, c_left, c_right]
        self.collision_detection = nn.Sequential(
            nn.Linear(4, 6),  # Input: [x', y', vx', vy']
            nn.ReLU(),
            nn.Linear(6, 3),  # Output: [c_ground, c_left, c_right]
            nn.Sigmoid()  # Collision probabilities
        )
        # Third hidden layer: Final state adjustment
        self.final_state = nn.Sequential(
            nn.Linear(7, 8),  # Input: [x', y', vx', vy', c_ground, c_left, c_right]
            nn.ReLU(),
            nn.Linear(8, 4)  # Output: [x(t+Δt), y(t+Δt), vx(t+Δt), vy(t+Δt)]
        )

    def forward(self, x):
        free_motion_state = self.free_motion(x)
        collision_flags = self.collision_detection(free_motion_state)
        combined = torch.cat((free_motion_state, collision_flags), dim=1)
        final_state = self.final_state(combined)
        return final_state

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ParticleNet().to(device)
model.load_state_dict(torch.load('models/particle_net.pth', map_location=device))
model.eval()
print('Loaded particle_net.pth for inference')

class NNParticle:
    def __init__(self, x, y, vx, vy, elasticity=0.5, radius=5):
        self.initial_velocity = np.array([vx, vy], dtype='float64')
        self.position = np.array([x, y], dtype='float64')
        self.velocity = np.array([vx, vy], dtype='float64')
        self.radius = radius
        self.elasticity = elasticity
        self.color = elasticity_to_color(self.elasticity, channel=1)
        # Simulation parameters for neural network input
        self.g = GRAVITY[1]  # Gravity magnitude (e.g., 9.81 m/s²)
        self.dt = DELTA_T  # Time step (e.g., 0.01 s)
        self.x_min = 0  # Left boundary (e.g., 0)
        self.x_max = WIDTH  # Right boundary (e.g., WIDTH)

    def update(self):
        # Prepare input vector for neural network
        input_vec = np.array([
            self.position[0],  # x
            self.position[1],  # y
            self.velocity[0],  # vx
            self.velocity[1],  # vy
            self.elasticity,   # e
            self.initial_velocity[0],  # vx0
            self.initial_velocity[1],  # vy0
            self.g,            # g
            self.dt,           # Δt
            self.x_min,        # x_min
            self.x_max         # x_max
        ], dtype='float32')

        # Convert to PyTorch tensor and move to device
        input_tensor = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)

        # Get predicted next state [x(t+Δt), y(t+Δt), vx(t+Δt), vy(t+Δt)]
        with torch.no_grad():
            predicted_state = model(input_tensor).squeeze(0).cpu().numpy()

        # Update position and velocity
        self.position = predicted_state[:2]  # [x, y]
        self.velocity = predicted_state[2:]  # [vx, vy]

    def draw(self, surface):
        draw_position = (np.float64(self.position[0]), np.float64(self.position[1] - self.radius))
        pygame.draw.circle(surface, self.color, draw_position, self.radius)
