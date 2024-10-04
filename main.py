from simulation import Simulation

if __name__ == "__main__":
    mode = 'training'
    particle_number = 75
    duration = None
    stop_velocity = 0.1

    simulation = Simulation(mode, particle_number=particle_number, duration=duration, stop_velocity=stop_velocity)
    simulation.run()