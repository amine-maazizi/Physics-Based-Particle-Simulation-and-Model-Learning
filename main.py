from simulation.simulation import Simulation

if __name__ == "__main__":
    simulation = Simulation(
        mode='testing',         # Set to 'training' for training mode
        particle_number=10,         # Single particle for simplicity, as per your neural network design
        nb_trials=100,            # Sufficient trials for robust dataset
        trial_duration=5.0,              # 5 seconds per trial
        trial_stop_velocity=0.1          # Stop when velocity < 0.1 m/s
    )
    simulation.run()