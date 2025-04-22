import pygame
import numpy as np
import random
import time

from simulation.config import *
from simulation.particle import Particle
from simulation.nn_particles import NNParticle


class Simulation:
    def __init__(self, mode='testing', particle_number=50, nb_trials=1000, trial_duration=10.0, trial_stop_velocity=None):
        pygame.init()
        pygame.display.set_caption("Particle Simulation")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.mode = mode
        self.particle_number = particle_number
        self.nb_trials = nb_trials
        self.trial_duration = trial_duration  # Duration in seconds
        self.trial_stop_velocity = trial_stop_velocity  # Velocity threshold to stop trial
        self.running = True

        if mode == 'training':
            self.dataset = []  # List to store trial data
        else:
            self.particles = self.create_particles(particle_number)
            self.ml_particles = self.create_ml_particles() if mode == 'testing' else []

    def create_particles(self, num_particles):
        particles = []
        for _ in range(num_particles):
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            vx = random.uniform(LOW_INITAL_VECOLITY, HIGH_INITIAL_VELOCITY)
            vy = random.uniform(LOW_INITAL_VECOLITY, HIGH_INITIAL_VELOCITY)
            elasticity = random.uniform(0.5, 0.9)  # Random elasticity in [0.5, 0.9]
            particles.append(Particle(x, y, vx, vy, elasticity=elasticity))
        return particles

    def create_ml_particles(self):
        ml_particles = []
        for particle in self.particles:
            ml_particle = NNParticle(
                particle.position[0],
                particle.position[1],
                particle.velocity[0],
                particle.velocity[1],
                particle.elasticity
            )
            ml_particles.append(ml_particle)
        return ml_particles

    def check_stop_conditions(self, trial_time, particles):
        # Check time-based condition
        if self.trial_duration is not None and trial_time >= self.trial_duration:
            return True

        # Check stop velocity condition
        if self.trial_stop_velocity is not None:
            if all(np.linalg.norm(particle.velocity) < self.trial_stop_velocity for particle in particles):
                return True

        return False

    def run_trial(self, trial_idx):
        # Initialize particles for the trial
        particles = self.create_particles(self.particle_number)
        trial_data = []  # Store states for this trial
        timestep = 0
        trial_start_time = time.time()

        pygame.display.set_caption(f"Trial {trial_idx + 1}/{self.nb_trials}")

        while True:
            trial_time = time.time() - trial_start_time

            # Check stop conditions
            if self.check_stop_conditions(trial_time, particles):
                break

            # Collect current state data for all particles (inputs)
            state = np.zeros((self.particle_number, 11))  # [x, y, vx, vy, elasticity, vx_initial, vy_initial, g, Δt, x_min, x_max]
            for i, particle in enumerate(particles):
                state[i] = [
                    particle.position[0],
                    particle.position[1],
                    particle.velocity[0],
                    particle.velocity[1],
                    particle.elasticity,
                    particle.initial_velocity[0],  # True initial vx
                    particle.initial_velocity[1],  # True initial vy
                    -9.81 * DELTA_T,              # Gravity y-component scaled by Δt
                    DELTA_T,                      # Timestep
                    0,                            # x_min
                    WIDTH                         # x_max
                ]

            trial_data.append(state)

            # Update particles
            for particle in particles:
                particle.update()

            timestep += 1

            # Handle events to prevent window freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False

            # Render (optional during training)
            self.screen.fill(BACKGROUND_COLOR)
            for particle in particles:
                particle.draw(self.screen)
            pygame.display.update()
            self.clock.tick(60)

        # Convert trial data to numpy array and store
        trial_data = np.array(trial_data)  # Shape: (T, n, 11)
        self.dataset.append(trial_data)
        print(f"Trial {trial_idx + 1}/{self.nb_trials} completed with {timestep} timesteps.")
        return True

    def run(self):
        if self.mode == 'training':
            # Run multiple trials
            for trial_idx in range(self.nb_trials):
                if not self.run_trial(trial_idx):
                    break
                # Reset for next trial
                self.screen.fill(BACKGROUND_COLOR)
                pygame.display.update()
            self.save_data()
        else:
            # Testing mode
            self.timestep = 0
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False

                # Update particles
                for particle in self.particles:
                    particle.update()
                if self.mode == 'testing':
                    for ml_particle in self.ml_particles:
                        ml_particle.update()

                self.timestep += 1

                # Render
                self.screen.fill(BACKGROUND_COLOR)
                for particle in self.particles:
                    particle.draw(self.screen)
                if self.mode == 'testing':
                    for ml_particle in self.ml_particles:
                        ml_particle.draw(self.screen)
                pygame.display.update()
                self.clock.tick(60)

        pygame.quit()

    def save_data(self):
        # Stack trial data into a tensor of shape (N, T_max, n, 11)
        max_timesteps = max(trial.shape[0] for trial in self.dataset)
        dataset_array = np.zeros((self.nb_trials, max_timesteps, self.particle_number, 11))
        for i, trial in enumerate(self.dataset):
            T = trial.shape[0]
            dataset_array[i, :T, :, :] = self.dataset[i]

        # Save as NumPy file
        np.save('particle_dataset.npy', dataset_array)
        print(f"Training dataset saved to 'particle_dataset.npy' with shape {dataset_array.shape}.")