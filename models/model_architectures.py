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
    
    def compute_physics_loss(self, pred, x):
        return 0.0  # Placeholder for physics loss computation

class ParticleNetRNN(nn.Module):
    """
    A Recurrent Neural Network (RNN) for predicting the next position of a single particle
    in a 2D environment, governed by Newtonian mechanics with gravity and elastic collisions
    with boundaries. The architecture uses a GRU to capture temporal dependencies in the
    particle's trajectory, followed by collision detection and final position adjustment.

    The input vector consists of 10 features:
    [x, y, v_x0, v_y0, g, Δt, e, x_min, x_max, n], representing the current position,
    initial velocities, gravitational acceleration, time step, coefficient of restitution,
    boundary positions, and step number (to compute time t = n * Δt).

    The output is the next position [x(t+Δt), y(t+Δt)].

    Attributes:
        gru (nn.GRU): Processes the input sequence to predict the free motion state.
        collision_detection (nn.Sequential): Detects collisions [c_ground, c_left, c_right]
            using the GRU output concatenated with e, x_min, and x_max.
        final_state (nn.Sequential): Adjusts the position for collisions, using the GRU output,
            collision flags, and e to produce the final next position.

    Args:
        input_size (int): Size of the input feature vector (default: 10).
        hidden_size (int): Size of the GRU hidden state (default: 16).
        num_layers (int): Number of GRU layers (default: 1).

    Methods:
        forward(x): Processes the input tensor through the GRU, collision detection, and
            final state adjustment to produce the next position.
    """
    def __init__(self, hidden_size=6, num_layers=1):
        super(ParticleNetRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layer to predict free motion state
        self.gru = nn.GRU(10, hidden_size, num_layers, batch_first=True)
        self.gru_output = nn.Linear(hidden_size, 2)  # Output: [x', y']

        # Collision detection layer
        self.collision_detection = nn.Sequential(
            nn.Linear(5, 6),  # Input: [x', y', e, x_min, x_max]
            nn.ReLU(),
            nn.Linear(6, 3),  # Output: [c_ground, c_left, c_right]
            nn.Sigmoid()      # Collision probabilities
        )

        # Final state adjustment layer
        self.final_state = nn.Sequential(
            nn.Linear(6, 8),  # Input: [x', y', c_ground, c_left, c_right, e]
            nn.ReLU(),
            nn.Linear(8, 2)   # Output: [x(t+Δt), y(t+Δt)]
        )

    def forward(self, x):
        """
        Forward pass through the RNN, processing the input sequence to predict the next position.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, 10), containing
                [x, y, v_x0, v_y0, g, Δt, e, x_min, x_max, n] for each time step.
                For a single time step, seq_len=1.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 2), containing the predicted
                next position [x(t+Δt), y(t+Δt)].
        """
        x = x.unsqueeze(1)  # From [batch_size, 10] to [batch_size, 1, 10]    

        # Extract relevant input features
        e = x[:, :, 6:7]       # [batch_size, seq_len, 1]
        bounds = x[:, :, 7:9]  # [batch_size, seq_len, 2]

        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # GRU forward pass
        gru_out, _ = self.gru(x, h0)  # gru_out: [batch_size, seq_len, hidden_size]
        # Take the output from the last time step
        gru_out = gru_out[:, -1, :]   # [batch_size, hidden_size]
        free_motion_state = self.gru_output(gru_out)  # [batch_size, 2]

        # Collision detection: Use the last time step's e and bounds
        collision_input = torch.cat((free_motion_state, e[:, -1, :], bounds[:, -1, :]), dim=1)  # [batch_size, 5]
        collision_flags = self.collision_detection(collision_input)  # [batch_size, 3]

        # Final state: Concatenate free motion state, collision flags, and e
        final_input = torch.cat((free_motion_state, collision_flags, e[:, -1, :]), dim=1)  # [batch_size, 6]
        final_state = self.final_state(final_input)  # [batch_size, 2]
        return final_state
    

class ParticleNetPINN(nn.Module):
    """
    A Physics-Informed Neural Network (PINN) for predicting the next position of a single particle
    in a 2D environment, governed by Newtonian mechanics with gravity and elastic collisions.
    The architecture extends ParticleNetSCFC, predicting the next position [x^{n+1}, y^{n+1}]
    and incorporating physics constraints via a composite loss function.

    The input vector consists of 10 features:
    [x, y, v_x0, v_y0, g, Δt, e, x_min, x_max, n], representing the current position,
    initial velocities, gravitational acceleration, time step, coefficient of restitution,
    boundary positions, and step number (to compute time t = n * Δt).

    The output is the next position [x^{n+1}, y^{n+1}].

    Attributes:
        free_motion (nn.Sequential): Predicts the free motion position [x', y']
            using the input vector.
        collision_detection (nn.Sequential): Detects collisions [c_ground, c_left, c_right]
            using the free motion position concatenated with e, x_min, and x_max.
        final_state (nn.Sequential): Adjusts the position for collisions, producing the final
            next position.

    Args:
        None

    Methods:
        forward(x): Processes the input tensor through the network to produce the next position.
        compute_physics_loss(pred, x): Computes the physics-based loss enforcing free motion
            and collision constraints.
    """
    def __init__(self):
        super(ParticleNetPINN, self).__init__()
        # Free motion subnetwork: Predict [x', y']
        self.free_motion = nn.Sequential(
            nn.Linear(10, 6),  # Input: [x, y, v_x0, v_y0, g, Δt, e, x_min, x_max, n]
            nn.ReLU(),
            nn.Linear(6, 2)    # Output: [x', y']
        )
        # Collision detection subnetwork
        self.collision_detection = nn.Sequential(
            nn.Linear(5, 6),   # Input: [x', y', e, x_min, x_max]
            nn.ReLU(),
            nn.Linear(6, 3),   # Output: [c_ground, c_left, c_right]
            nn.Sigmoid()       # Collision probabilities
        )
        # Final state subnetwork
        self.final_state = nn.Sequential(
            nn.Linear(6, 8),   # Input: [x', y', c_ground, c_left, c_right, e]
            nn.ReLU(),
            nn.Linear(8, 2)    # Output: [x^{n+1}, y^{n+1}]
        )

    def forward(self, x):
        """
        Forward pass through the network to predict the next position.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 10), containing
                [x, y, v_x0, v_y0, g, Δt, e, x_min, x_max, n].

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 2), containing the predicted
                next position [x^{n+1}, y^{n+1}].
        """
        # Extract relevant input features
        e = x[:, 6:7]          # [batch_size, 1]
        bounds = x[:, 7:9]     # [batch_size, 2]

        # Predict free motion position
        free_motion_state = self.free_motion(x)  # [batch_size, 2]

        # Collision detection
        collision_input = torch.cat((free_motion_state, e, bounds), dim=1)  # [batch_size, 5]
        collision_flags = self.collision_detection(collision_input)  # [batch_size, 3]

        # Final state prediction
        final_input = torch.cat((free_motion_state, collision_flags, e), dim=1)  # [batch_size, 6]
        next_position = self.final_state(final_input)  # [batch_size, 2]
        return next_position

    def compute_physics_loss(self, pred, x):
        """
        Compute the physics-informed loss, enforcing free motion dynamics and collision constraints.

        Args:
            pred (torch.Tensor): Predicted next position [x^{n+1}, y^{n+1}]
                of shape (batch_size, 2).
            x (torch.Tensor): Input tensor of shape (batch_size, 10), containing
                [x, y, v_x0, v_y0, g, Δt, e, x_min, x_max, n].

        Returns:
            torch.Tensor: Physics loss scalar.
        """
        # Extract input features
        x_n = x[:, 0]          # Current x position
        y_n = x[:, 1]          # Current y position
        v_x0 = x[:, 2]         # Initial x velocity (v_x0)
        v_y0 = x[:, 3]         # Initial y velocity (v_y0)
        g = x[:, 4]            # Gravity
        dt = x[:, 5]           # Time step
        e = x[:, 6]            # Coefficient of restitution
        x_min = x[:, 7]        # Left boundary
        x_max = x[:, 8]        # Right boundary
        n = x[:, 9]            # Step number

        # Predicted next position
        x_np1 = pred[:, 0]     # Predicted x^{n+1}
        y_np1 = pred[:, 1]     # Predicted y^{n+1}

        # Compute current velocities using initial velocities and time
        # v_x(t) = v_x0, v_y(t) = v_y0 - g * t, where t = n * Δt
        t = n * dt
        v_x_n = v_x0
        v_y_n = v_y0 - g * t

        # Free motion physics loss (Euler integration)
        # Expected: x^{n+1} = x^n + v_x^n * Δt
        #           y^{n+1} = y^n + v_y^n * Δt - 0.5 * g * Δt^2
        loss_x = torch.mean((x_np1 - (x_n + v_x_n * dt))**2)
        loss_y = torch.mean((y_np1 - (y_n + v_y_n * dt - 0.5 * g * dt**2))**2)
        loss_free_motion = loss_x + loss_y

        # Collision loss
        loss_collision = torch.zeros(1, device=x.device)
        # Ground collision: y^{n+1} <= 0 and v_y^n < 0
        ground_mask = (y_np1 <= 0) & (v_y_n < 0)
        if ground_mask.any():
            # Approximate v_y^{n+1} using finite difference: (y^{n+1} - y^n) / Δt
            v_y_np1 = (y_np1[ground_mask] - y_n[ground_mask]) / dt[ground_mask]
            v_y_before = v_y_n[ground_mask]
            e_ground = e[ground_mask]
            loss_ground = torch.mean((v_y_np1 + e_ground * v_y_before)**2)
            loss_collision += loss_ground
            # Penalize y^{n+1} < 0
            loss_collision += torch.mean(y_np1[ground_mask]**2)

        # Left wall collision: x^{n+1} <= x_min and v_x^n < 0
        left_mask = (x_np1 <= x_min) & (v_x_n < 0)
        if left_mask.any():
            # Approximate v_x^{n+1} using finite difference: (x^{n+1} - x^n) / Δt
            v_x_np1 = (x_np1[left_mask] - x_n[left_mask]) / dt[left_mask]
            v_x_before = v_x_n[left_mask]
            e_left = e[left_mask]
            loss_left = torch.mean((v_x_np1 + e_left * v_x_before)**2)
            loss_collision += loss_left
            loss_collision += torch.mean((x_np1[left_mask] - x_min[left_mask])**2)

        # Right wall collision: x^{n+1} >= x_max and v_x^n > 0
        right_mask = (x_np1 >= x_max) & (v_x_n > 0)
        if right_mask.any():
            v_x_np1 = (x_np1[right_mask] - x_n[right_mask]) / dt[right_mask]
            v_x_before = v_x_n[right_mask]
            e_right = e[right_mask]
            loss_right = torch.mean((v_x_np1 + e_right * v_x_before)**2)
            loss_collision += loss_right
            loss_collision += torch.mean((x_np1[right_mask] - x_max[right_mask])**2)

        # Combine losses with weights
        lambda_free = 1.0
        lambda_collision = 1.0
        total_physics_loss = lambda_free * loss_free_motion + lambda_collision * loss_collision
        return total_physics_loss
    

# Example training loop (for reference)
"""
# Assuming dataset is loaded as x_train, y_train (shapes: [N, 11], [N, 4])
model = ParticleNetPINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(x_train), batch_size):
        batch_x = x_train[i:i+batch_size]  # [batch_size, 11]
        batch_y = y_train[i:i+batch_size]  # [batch_size, 4]

        # Forward pass
        pred = model(batch_x)
        data_loss = criterion(pred, batch_y)
        physics_loss = model.compute_physics_loss(pred, batch_x)
        total_loss = data_loss + 0.1 * physics_loss  # Adjust weight as needed

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
"""