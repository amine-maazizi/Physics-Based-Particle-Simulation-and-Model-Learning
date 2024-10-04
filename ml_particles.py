import joblib  # For loading the ML model
from keras.models import load_model
import numpy as np
import pygame
from config import *
from utils import elasticity_to_color

class MLParticle:
    def __init__(self, x, y, vx, vy, elasticity):
        self.position = np.array([x, y], dtype='float64')
        self.velocity = np.array([vx, vy], dtype='float64') 
        
        self.elasticity = elasticity  
        self.gravity_x = GRAVITY[0]
        self.gravity_y = GRAVITY[1]
        self.damping = DAMPING_FACTOR
        
        self.model = load_model('nn_model.h5', compile=False)
        
        self.color = elasticity_to_color(elasticity, 1)

    def predict_next_position(self, timestep):
        features = np.array([
            self.position[0],
            self.position[1],
            self.velocity[0],
            self.velocity[1],
            timestep,
            self.elasticity,
        ]).reshape(1, -1)

        predicted_position = self.model.predict(features)[0]
        return predicted_position

    def update(self, timestep):
        self.position = self.predict_next_position(timestep)

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, self.position.astype(int), 5) 
