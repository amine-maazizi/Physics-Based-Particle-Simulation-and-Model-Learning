import pygame
import numpy as np
import random

from config import *
from utils import elasticity_to_color


class Particle:
    def __init__(self, x, y, vx, vy, radius=5):
        self.position = np.array([x, y], dtype='float64')
        self.velocity = np.array([vx, vy], dtype='float64')
        self.radius = radius
        self.elasticity = random.uniform(LOW_ELASTICITY, HIGH_ELASTICITY)  # Random elasticity
        self.color = elasticity_to_color(self.elasticity)  # Set initial color based on elasticity

    def update(self):
        # Apply gravity
        self.velocity += GRAVITY
        
        # Apply damping
        self.velocity *= DAMPING_FACTOR
        
        # Update position
        self.position += self.velocity

        # Ensure the particle does not go below the ground
        if self.position[1] >= (HEIGHT - self.radius):
            self.position[1] = HEIGHT - self.radius  # Reset position to ground level
            self.velocity[1] *= -self.elasticity  # Reverse velocity with elasticity
            self.color = elasticity_to_color(self.elasticity)  # Update color on bounce
        
        # Bounce off left and right walls
        if self.position[0] <= self.radius or self.position[0] >= (WIDTH - self.radius):
            self.velocity[0] *= -self.elasticity  # Reverse velocity with elasticity
            self.color = elasticity_to_color(self.elasticity)  # Update color on bounce

    def draw(self, surface):
        # Draw the particle with its position adjusted for the radius
        draw_position = (self.position[0], self.position[1] - self.radius)  # Center the position at the bottom
        pygame.draw.circle(surface, self.color, draw_position, self.radius)