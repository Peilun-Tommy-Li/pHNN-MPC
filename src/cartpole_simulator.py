"""
Simple cart-pole simulator for MPC testing.

Does not require gym/gymnasium - pure NumPy implementation.
"""

import numpy as np


class CartPoleSimulator:
    """
    Simple cart-pole dynamics simulator.

    State: [x, theta, x_dot, theta_dot]
    Control: [force]
    """

    def __init__(self, dt=0.02):
        """
        Initialize cart-pole simulator.

        Args:
            dt: Time step size
        """
        self.dt = dt

        # Physical parameters (standard cart-pole)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 0.5  # Half-length of pole
        self.polemass_length = self.masspole * self.length

        # Derived parameters
        self.total_mass = self.masspole + self.masscart

        # State
        self.state = None

    def reset(self, initial_state=None):
        """
        Reset simulator to initial state.

        Args:
            initial_state: [x, theta, x_dot, theta_dot] or None for default

        Returns:
            state: Initial state
        """
        if initial_state is None:
            # Small random perturbation from upright
            self.state = np.array([
                0.0,  # x
                np.random.uniform(-0.05, 0.05),  # theta
                0.0,  # x_dot
                0.0   # theta_dot
            ])
        else:
            self.state = np.array(initial_state, dtype=np.float64)

        return self.state.copy()

    def step(self, action):
        """
        Take one simulation step.

        Args:
            action: Control force (scalar or array with single element)

        Returns:
            state: New state [x, theta, x_dot, theta_dot]
            done: Whether episode is done
        """
        # Extract action
        if isinstance(action, np.ndarray):
            force = action[0]
        else:
            force = action

        # Current state
        x, theta, x_dot, theta_dot = self.state

        # Dynamics (standard cart-pole equations)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # Temporary variable
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass

        # Angular acceleration
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))

        # Linear acceleration
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration
        x = x + self.dt * x_dot
        theta = theta + self.dt * theta_dot
        x_dot = x_dot + self.dt * xacc
        theta_dot = theta_dot + self.dt * thetaacc

        # Update state
        self.state = np.array([x, theta, x_dot, theta_dot])

        # Check termination (pole falls over or cart goes too far)
        done = bool(
            abs(x) > 10.0 or  # Cart position limit
            abs(theta) > 0.5  # Pole angle limit (about 30 degrees)
        )

        return self.state.copy(), done

    def get_state(self):
        """Get current state."""
        return self.state.copy()
