import pygame
import numpy as np
import random
import time  

import pandas as pd  

from config import *
from particle import Particle
from ml_particles import MLParticle 


class Simulation:
    def __init__(self, mode='testing', particle_number=50, duration=None, stop_velocity=None):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.mode = mode
        self.running = True
        
        self.particles = self.create_particles(particle_number)  # Create physics-based particles
        if mode == 'testing':
            self.ml_particles = self.create_ml_particles()  
        
        if mode == 'training':
            self.data_set = []  
        
        # Stop conditions
        self.duration = duration 
        self.stop_velocity = stop_velocity 
        
        # Time tracking
        self.start_time = time.time()  
        self.timestep = 0  

    def create_particles(self, num_particles):
        particles = []
        for _ in range(num_particles):
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            vx = random.uniform(LOW_INITAL_VECOLITY, HIGH_INITIAL_VELOCITY)
            vy = random.uniform(LOW_INITAL_VECOLITY, HIGH_INITIAL_VELOCITY)
            particles.append(Particle(x, y, vx, vy))
        return particles

    def create_ml_particles(self):
        ml_particles = []
        for particle in self.particles:
            # Spawn ML particles at the same positions and with the same velocities as the regular particles
            ml_particle = MLParticle(
                particle.position[0],
                particle.position[1],
                particle.velocity[0],
                particle.velocity[1],
                particle.elasticity  
            )
            ml_particles.append(ml_particle)
        return ml_particles

    def physics_process(self):
        for particle in self.particles:
            # Check stop conditions
            if self.check_stop_conditions():
                self.running = False
                break  

            if self.mode == 'training':
                self.collect_data(particle)

            particle.update()

        if self.mode == 'testing':
            for ml_particle in self.ml_particles:
                ml_particle.update(self.timestep)  # Update ML-based particle position

        self.timestep += 1  # Increment the time step counter

    def collect_data(self, particle):
        # Collect current state data
        current_data = {
            'time_step': self.timestep,  # Store the current time step
            'current_x': particle.position[0],
            'current_y': particle.position[1],
            'current_vx': particle.velocity[0],
            'current_vy': particle.velocity[1],
            'elasticity': particle.elasticity
        }
        
        # Update particle position before collecting next state data
        particle.update()
        
        # Collect next state data
        next_data = {
            'next_x': particle.position[0],
            'next_y': particle.position[1]
        }

        # Combine current and next data
        full_data = {**current_data, **next_data}
        self.data_set.append(full_data)

    def check_stop_conditions(self):
        # Check time-based condition
        if self.duration is not None and (time.time() - self.start_time) >= self.duration:
            return True
        
        # Check stop velocity condition, if defined
        if self.stop_velocity is not None:
            if all(np.linalg.norm(particle.velocity) < self.stop_velocity for particle in self.particles):
                return True
        
        return False

    def render(self):
        self.screen.fill(BACKGROUND_COLOR)
        for particle in self.particles:
            particle.draw(self.screen)  
        if self.mode == 'testing':
            for ml_particle in self.ml_particles:
                ml_particle.draw(self.screen)  
        pygame.display.update()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.physics_process()
            self.render()
            self.clock.tick(60) 

        pygame.quit()
        
        # Saving dataset
        if self.mode == 'training':
            self.save_data()

    def save_data(self):
        df = pd.DataFrame(self.data_set)
        df.to_csv('particle_data.csv', index=False)
        print("Training data saved to 'particle_data.csv'.")
