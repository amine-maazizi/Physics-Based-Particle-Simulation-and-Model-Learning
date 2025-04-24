import torch
import torch.nn as nn


# Neural Network Architecture
class ParticleNet(nn.Module):
    def __init__(self):
        super(ParticleNet, self).__init__()
        # First hidden layer: Predict free motion state [x', y', vx', vy']
        # 6 neurons: 2 for x', 3 for y', 1 for vx', 1 for vy'
        self.free_motion = nn.Sequential(
            nn.Linear(11, 6),  # Input: [x, y, vx, vy, e, vx0, vy0, g, Δt, x_min, x_max]
            nn.ReLU(),
            nn.Linear(6, 4)  # Output: [x', y', vx', vy']
        )
        # Second hidden layer: Predict collision flags [c_ground, c_left, c_right]
        # 6 neurons: 2 per collision flag
        self.collision_detection = nn.Sequential(
            nn.Linear(4, 6),  # Input: [x', y', vx', vy']
            nn.ReLU(),
            nn.Linear(6, 3),  # Output: [c_ground, c_left, c_right]
            nn.Sigmoid()  # Collision probabilities
        )
        # Third hidden layer: Final state adjustment
        # 8 neurons: 2 per output (x, y, vx, vy)
        self.final_state = nn.Sequential(
            nn.Linear(7, 8),  # Input: [x', y', vx', vy', c_ground, c_left, c_right]
            nn.ReLU(),
            nn.Linear(8, 4)  # Output: [x(t+Δt), y(t+Δt), vx(t+Δt), vy(t+Δt)]
        )

    def forward(self, x):
        # Free motion prediction
        free_motion_state = self.free_motion(x)  # [x', y', vx', vy']
        # Collision detection
        collision_flags = self.collision_detection(free_motion_state)  # [c_ground, c_left, c_right]
        # Concatenate free motion state and collision flags
        combined = torch.cat((free_motion_state, collision_flags), dim=1)
        # Final state prediction
        final_state = self.final_state(combined)  # [x(t+Δt), y(t+Δt), vx(t+Δt), vy(t+Δt)]
        return final_state
    

class ParticleNetSCFC(nn.Module):
    """
    A neural network for predicting the next state of a single particle in a 2D environment,
    governed by Newtonian mechanics with gravity and elastic collisions with boundaries.
    The architecture is designed to model free motion and collision dynamics, using a modular
    structure with skip connections and feature concatenation to ensure effective utilization
    of input features across layers.

    The input vector consists of 11 features:
    [x, y, vx, vy, e, vx0, vy0, g, Δt, x_min, x_max], representing the current position,
    velocity, coefficient of restitution, initial velocities, gravitational acceleration,
    time step, and boundary positions. The output is the next state [x(t+Δt), y(t+Δt),
    vx(t+Δt), vy(t+Δt)].

    The architecture addresses the limitation of underutilized input features (e.g., e, x_min,
    x_max) in earlier layers by incorporating skip connections and feature concatenation:
    - Skip connections pass critical features (e, x_min, x_max) directly to the collision
      detection and final state layers, bypassing earlier transformations.
    - Feature concatenation combines these inputs with intermediate outputs (e.g., free motion
      state, collision flags), ensuring that physically relevant features are available where
      needed (e.g., e for collision response, x_min/x_max for boundary detection).

    Attributes:
        free_motion (nn.Sequential): Predicts the free motion state [x', y', vx', vy']
            using the full input vector.
        collision_detection (nn.Sequential): Detects collisions [c_ground, c_left, c_right]
            using the free motion state concatenated with e, x_min, and x_max.
        final_state (nn.Sequential): Adjusts the state for collisions, using the free motion
            state, collision flags, and e to produce the final next state.

    Args:
        None

    Methods:
        forward(x): Processes the input tensor through the network, applying skip connections
            and feature concatenation to produce the next state.
    """
    def __init__(self):
        super(ParticleNetSCFC, self).__init__()
        # First hidden layer: Predict free motion state [x', y', vx', vy']
        self.free_motion = nn.Sequential(
            nn.Linear(8, 6),  # Input: [x, y, vx, vy, e, vx0, vy0, g, Δt, x_min, x_max]
            nn.ReLU(),
            nn.Linear(6, 4)  # Output: [x', y', vx', vy']
        )
        # Second hidden layer: Collision detection
        self.collision_detection = nn.Sequential(
            nn.Linear(7, 6),  # Input: [x', y', vx', vy', e, x_min, x_max]
            nn.ReLU(),
            nn.Linear(6, 3),  # Output: [c_ground, c_left, c_right]
            nn.Sigmoid()  # Collision probabilities
        )
        # Third hidden layer: Final state adjustment
        self.final_state = nn.Sequential(
            nn.Linear(8, 8),  # Input: [x', y', vx', vy', c_ground, c_left, c_right, e]
            nn.ReLU(),
            nn.Linear(8, 4)  # Output: [x(t+Δt), y(t+Δt), vx(t+Δt), vy(t+Δt)]
        )

    def forward(self, x):
        """
        Forward pass through the network, applying skip connections and feature concatenation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 11), containing
                [x, y, vx, vy, e, vx0, vy0, g, Δt, x_min, x_max].

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 4), containing the predicted
                next state [x(t+Δt), y(t+Δt), vx(t+Δt), vy(t+Δt)].
        """
        # Extract relevant input features
        state = x[:, :4]  # [x, y, vx, vy]
        e = x[:, 4:5]     # [e]
        bounds = x[:, 9:11]  # [x_min, x_max]
        
        # Free motion prediction
        x = torch.cat((state, x[:, 5:9]), dim=1)  # Concatenate with [vx0, vy0, g, Δt]
        free_motion_state = self.free_motion(x)  # [x', y', vx', vy']
        
        # Collision detection: Concatenate free motion state with e, x_min, x_max
        collision_input = torch.cat((free_motion_state, e, bounds), dim=1)  # [x', y', vx', vy', e, x_min, x_max]
        collision_flags = self.collision_detection(collision_input)  # [c_ground, c_left, c_right]
        
        # Final state: Concatenate free motion state, collision flags, and e
        final_input = torch.cat((free_motion_state, collision_flags, e), dim=1)  # [x', y', vx', vy', c_ground, c_left, c_right, e]
        final_state = self.final_state(final_input)  # [x(t+Δt), y(t+Δt), vx(t+Δt), vy(t+Δt)]
        
        return final_state