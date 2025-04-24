import torch
import torch.nn as nn

# Neural Network Architecture
class ParticleNet(nn.Module):
    def __init__(self):
        super(ParticleNet, self).__init__()
        # First hidden layer: Predict free motion position [x', y']
        self.free_motion = nn.Sequential(
            nn.Linear(10, 6),  # Input: [x, y, v_x0, v_y0, g, Δt, e, x_min, x_max, n]
            nn.ReLU(),
            nn.Linear(6, 2)   # Output: [x', y']
        )
        # Second hidden layer: Predict collision flags [c_ground, c_left, c_right]
        self.collision_detection = nn.Sequential(
            nn.Linear(5, 6),  # Input: [x', y', e, x_min, x_max]
            nn.ReLU(),
            nn.Linear(6, 3),  # Output: [c_ground, c_left, c_right]
            nn.Sigmoid()      # Collision probabilities
        )
        # Third hidden layer: Final position adjustment
        self.final_state = nn.Sequential(
            nn.Linear(6, 8),  # Input: [x', y', c_ground, c_left, c_right, e]
            nn.ReLU(),
            nn.Linear(8, 2)   # Output: [x(t+Δt), y(t+Δt)]
        )

    def forward(self, x):
        # Extract features for collision detection
        e = x[:, 6:7]          # [e]
        bounds = x[:, 7:9]     # [x_min, x_max]

        # Predict free motion position directly from input
        free_motion_state = self.free_motion(x)  # [x', y']

        # Collision detection
        collision_input = torch.cat((free_motion_state, e, bounds), dim=1)
        collision_flags = self.collision_detection(collision_input)  # [c_ground, c_left, c_right]

        # Concatenate free motion state and collision flags
        combined = torch.cat((free_motion_state, collision_flags, e), dim=1)
        # Final state prediction
        final_state = self.final_state(combined)  # [x(t+Δt), y(t+Δt)]
        return final_state

class ParticleNetSCFC(nn.Module):
    """
    A neural network for predicting the next position of a single particle in a 2D environment,
    governed by Newtonian mechanics with gravity and elastic collisions with boundaries.
    The architecture uses skip connections and feature concatenation to ensure effective utilization
    of input features across layers.

    The input vector consists of 10 features:
    [x, y, v_x0, v_y0, g, Δt, e, x_min, x_max, n], representing the current position,
    initial velocities, gravitational acceleration, time step, coefficient of restitution,
    boundary positions, and step number (to compute time t = n * Δt).

    The output is the next position [x(t+Δt), y(t+Δt)].

    Attributes:
        free_motion (nn.Sequential): Predicts the free motion position [x', y']
            using the input vector.
        collision_detection (nn.Sequential): Detects collisions [c_ground, c_left, c_right]
            using the free motion position concatenated with e, x_min, and x_max.
        final_state (nn.Sequential): Adjusts the position for collisions, using the free motion
            position, collision flags, and e to produce the final next position.

    Args:
        None

    Methods:
        forward(x): Processes the input tensor through the network, applying skip connections
            and feature concatenation to produce the next position.
    """
    def __init__(self):
        super(ParticleNetSCFC, self).__init__()
        # First hidden layer: Predict free motion position [x', y']
        self.free_motion = nn.Sequential(
            nn.Linear(10, 6),  # Input: [x, y, v_x0, v_y0, g, Δt, e, x_min, x_max, n]
            nn.ReLU(),
            nn.Linear(6, 2)   # Output: [x', y']
        )
        # Second hidden layer: Collision detection
        self.collision_detection = nn.Sequential(
            nn.Linear(5, 6),  # Input: [x', y', e, x_min, x_max]
            nn.ReLU(),
            nn.Linear(6, 3),  # Output: [c_ground, c_left, c_right]
            nn.Sigmoid()      # Collision probabilities
        )
        # Third hidden layer: Final position adjustment
        self.final_state = nn.Sequential(
            nn.Linear(6, 8),  # Input: [x', y', c_ground, c_left, c_right, e]
            nn.ReLU(),
            nn.Linear(8, 2)   # Output: [x(t+Δt), y(t+Δt)]
        )

    def forward(self, x):
        """
        Forward pass through the network, applying skip connections and feature concatenation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 10), containing
                [x, y, v_x0, v_y0, g, Δt, e, x_min, x_max, n].

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 2), containing the predicted
                next position [x(t+Δt), y(t+Δt)].
        """
        # Extract relevant input features
        e = x[:, 6:7]          # [e]
        bounds = x[:, 7:9]     # [x_min, x_max]

        # Predict free motion position directly from input
        free_motion_state = self.free_motion(x)  # [x', y']

        # Collision detection: Concatenate free motion state with e, x_min, x_max
        collision_input = torch.cat((free_motion_state, e, bounds), dim=1)  # [x', y', e, x_min, x_max]
        collision_flags = self.collision_detection(collision_input)  # [c_ground, c_left, c_right]

        # Final state: Concatenate free motion state, collision flags, and e
        final_input = torch.cat((free_motion_state, collision_flags, e), dim=1)  # [x', y', c_ground, c_left, c_right, e]
        final_state = self.final_state(final_input)  # [x(t+Δt), y(t+Δt)]
        return final_state