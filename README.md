# Physics-Based Particle Simulation and Model Learning

## Overview

This project simulates falling particles with wall and ground collision (no inter-particle collision). The generated simulation data is used to train two models: 
1. [x] **Linear Regression**
2. [x] **Neural Network**
3. [] **ARIMA**
4. [] **RNN**
5. [] **LSTM**

The goal is for the models to learn the physics governing the particle motion. In testing mode, both the actual simulation and the model-driven particles are rendered for comparison.

## Features

- **Particle Simulation:** Simulates falling particles with wall and ground collision.
- **Data Generation:** Exports simulation data to a CSV file.
- **Model Training:** Trains Linear Regression and Neural Network models using the exported simulation data.
- **Model Testing:** Renders both the physics simulation and model-predicted particles to visually verify model learning.
