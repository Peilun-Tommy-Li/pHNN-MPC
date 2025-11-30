import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import yaml
from src.pHNN import pHNN
from src.TrajectoryStepDataset import TrajectoryStepDataset


# -------------------------------
# Load config
# -------------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------------
# Save trained pHNN model
# -------------------------------
def save_model(model, path="trained_pHNN.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")


# -------------------------------
# Pendulum dynamics (ground truth ODE)
# -------------------------------
def pendulum_dynamics(x, u, params):
    m, l, g, b = params["m"], params["l"], params["g"], params["b"]
    theta, omega = x
    dtheta = omega
    domega = -(g / l) * np.sin(theta) - (b / (m * l ** 2)) * omega + u / (m * l ** 2)
    return np.array([dtheta, domega])


# -------------------------------
# Generate dataset
# Generate 'num_traj' many time-series dynamics of the same system
# Each time series is of length T/dt
# Derivative data is not necessary
# -------------------------------
def generate_data(config):
    pend_params = config["pendulum"]
    dt, T = pend_params["dt"], pend_params["T"]
    num_traj = pend_params["num_traj"]
    u_min, u_max = pend_params["u_min"], pend_params["u_max"]
    timesteps = int(T / dt)

    # Use master lists to store each trajectory
    all_states, all_inputs, all_derivatives = [], [], []

    for _ in range(num_traj):
        # random initial state
        theta0 = np.random.uniform(-np.pi, np.pi)
        omega0 = np.random.uniform(-1.0, 1.0)
        x = np.array([theta0, omega0])

        # Use temporary lists for the current trajectory
        traj_states, traj_inputs, traj_derivatives = [], [], []

        for _ in range(timesteps):
            u = np.random.uniform(u_min, u_max)  # random torque
            dx = pendulum_dynamics(x, u, pend_params)

            traj_states.append(x)
            traj_inputs.append([u])
            traj_derivatives.append(dx)

            # Euler integration
            x = x + dt * dx

        # Append the completed trajectory to the master lists
        all_states.append(traj_states)
        all_inputs.append(traj_inputs)
        all_derivatives.append(traj_derivatives)

    # Convert the lists of trajectories into 3D tensors
    return (
        torch.tensor(np.array(all_states), dtype=torch.float32),
        torch.tensor(np.array(all_inputs), dtype=torch.float32),
        torch.tensor(np.array(all_derivatives), dtype=torch.float32),
    )


# -------------------------------
# Training loop
# Train pHNN model: (X[i], U[i]) -> pHNN -> X[i+1]
# Derivative is NOT needed in training.
# -------------------------------
def train_pHNN_ode(model, config, loader):
    # param unpack
    lr = config["training"]["lr"]
    epochs = config["training"]["epochs"]
    dt = config["pendulum"]["dt"]

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        total_loss_dx = 0
        for x_batch, u_batch, dx_batch in loader:
            optimizer.zero_grad()

            # Correctly get the initial states for the entire batch
            # x_batch shape: (batch_size, seq_len, state_dim)
            x0_batch = x_batch[:, 0, :].requires_grad_(True)

            # Predict the rest of the sequence in a batched manner
            X_pred = [x0_batch]
            dX_pred = []

            for t in range(x_batch.shape[1] - 1):  # Correct loop range: iterate over seq_len
                # Pass the predicted state and the corresponding input for the batch
                # X_pred[-1] has shape (batch_size, state_dim)
                # u_batch[:, t, :] has shape (batch_size, input_dim)
                dx, _ = model(X_pred[-1], u_batch[:, t, :])
                dX_pred.append(dx)
                X_pred.append(X_pred[-1] + dt * dx)

            # Stack the list of tensors into a single tensor
            X_pred = torch.stack(X_pred, dim=1)  # Shape: (batch_size, seq_len, state_dim)
            dX_pred = torch.stack(dX_pred, dim=1)  # Shape: (batch_size, seq_len - 1, state_dim)

            # Compute losses
            loss = loss_fn(X_pred, x_batch)

            # Note: The dX_pred tensor has `seq_len - 1` points,
            # so we must compare it to the corresponding part of dx_batch.
            loss_dx = loss_fn(dX_pred, dx_batch[:, 0:-1, :])

            total_loss_dx += loss_dx.item()

            # Combine losses if you wish, or backpropagate them separately
            combined_loss = loss + loss_dx
            combined_loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Trajectory Loss: {total_loss / len(loader):.6f}")
        print(f"Epoch {epoch + 1}/{epochs} - dX Loss: {total_loss_dx / len(loader):.6f}")

    return model


# -------------------------------
# Evaluation
# randomly generate a trajectory given random control input
# compare pHNN model with this random trajectory
# -------------------------------
def evaluate(model, config, num_traj=1):
    """
    Evaluate the learned pHNN on full trajectories
    """
    pend_params = config["pendulum"]
    dt, T = pend_params["dt"], pend_params["T"]
    timesteps = int(T / dt)

    # Generate a ground-truth trajectory for comparison
    theta0 = np.random.uniform(-np.pi, np.pi)
    omega0 = np.random.uniform(-1.0, 1.0)
    x_gt = np.array([theta0, omega0])
    X_true, U_true = [x_gt.copy()], []

    for t in range(timesteps):
        u = np.random.uniform(pend_params["u_min"], pend_params["u_max"])
        # u=0
        dx = pendulum_dynamics(x_gt, u, pend_params)
        x_gt = x_gt + dt * dx

        X_true.append(x_gt.copy())
        U_true.append([u])

    X_true = torch.tensor(np.array(X_true), dtype=torch.float32)
    U_true = torch.tensor(np.array(U_true), dtype=torch.float32)

    # Integrate the pHNN to get predicted trajectory
    # X_pred = [X_true[0]]
    X_pred = [X_true[0].unsqueeze(0).requires_grad_(True)]

    for t in range(1, X_true.shape[0]):
        dx, _ = model(X_pred[-1].unsqueeze(0), U_true[t - 1].unsqueeze(0))
        X_pred.append(X_pred[-1] + dt * dx)
    X_pred = torch.cat(X_pred, dim=0)

    # Compute trajectory MSE
    loss = nn.MSELoss()(X_pred, X_true).item()
    print("Trajectory MSE:", loss)

    # Plot
    plt.plot(X_true[:, 0].numpy(), label="True θ")
    plt.plot(X_pred[:, 0].detach().numpy(), "--", label="Pred θ")
    plt.plot(X_true[:, 1].numpy(), label="True ω")
    plt.plot(X_pred[:, 1].detach().numpy(), "--", label="Pred ω")
    plt.legend()
    plt.show()


def compare_learned_components(model, config):
    """
    Compares the learned pHNN components (J, R, H, G) with their ground truth.

    Args:
        model (nn.Module): The trained pHNN model.
        config (dict): The configuration dictionary.
    """
    pend_params = config["pendulum"]
    state_dim = config["model"]["state_dim"]
    input_dim = config["model"]["input_dim"]

    model.eval()

    #################################################################
    # 1. Corrected Ground Truth Functions and Matrices for Pendulum
    #################################################################

    m, g, L = pend_params["m"], pend_params["g"], pend_params["l"]
    b = pend_params["b"]

    def H_true_fn(x):
        theta, omega = x[:, 0], x[:, 1]
        return 0.5 * m * (L * omega) ** 2 + m * g * L * (1 - torch.cos(theta))

    def R_true_fn(x):
        return torch.tensor([[[0.0, 0.0], [0.0, b]] for _ in range(x.shape[0])])

    def G_true_fn(x):
        return torch.tensor([[[0.0], [1.0]] for _ in range(x.shape[0])])

    J_true = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32)

    #################################################################
    # 2. J Matrix Comparison
    #################################################################

    print("--- J Matrix Comparison ---")
    learned_J = (model.J.data - model.J.data.T) / 2
    diff_J = torch.norm(learned_J - J_true)

    print("Learned J:\n", learned_J.numpy())
    print("Ground Truth J:\n", J_true.numpy())
    print(f"Frobenius Norm of Difference: {diff_J.item():.6f}")

    #################################################################
    # 3. State-Dependent Components (H, R, G) Visualization
    #################################################################

    print("\n--- Visualizing State-Dependent Functions ---")

    theta_range = np.linspace(-np.pi, np.pi, 100)
    omega_range = np.linspace(-3.0, 3.0, 100)
    grid_theta, grid_omega = np.meshgrid(theta_range, omega_range)
    grid_states = torch.tensor(np.stack([grid_theta.flatten(), grid_omega.flatten()], axis=1), dtype=torch.float32)

    with torch.no_grad():
        learned_H_vals = model.H_net(grid_states).numpy().reshape(grid_theta.shape)
        true_H_vals = H_true_fn(grid_states).numpy().reshape(grid_theta.shape)
        mse_H = np.mean((learned_H_vals - true_H_vals) ** 2)

        r_out = model.R_net(grid_states)
        learned_R_vals_raw = r_out.numpy().reshape(-1, state_dim, state_dim)
        learned_R_vals = learned_R_vals_raw @ learned_R_vals_raw.transpose(0, 2, 1)

        true_R_vals = R_true_fn(grid_states).numpy()
        mse_R = np.mean((learned_R_vals - true_R_vals) ** 2)

        learned_G_vals = model.G_net(grid_states).numpy()
        true_G_vals = G_true_fn(grid_states).numpy()
        true_G_vals = np.squeeze(true_G_vals, axis=2)
        mse_G = np.mean((learned_G_vals - true_G_vals) ** 2)

    print(f"Mean Squared Error (H): {mse_H:.6f}")
    print(f"Mean Squared Error (R): {mse_R:.6f}")
    print(f"Mean Squared Error (G): {mse_G:.6f}")

    fig, axs = plt.subplots(2, 3, figsize=(16, 10))

    im1 = axs[0, 0].contourf(grid_theta, grid_omega, learned_H_vals, 20, cmap='viridis')
    fig.colorbar(im1, ax=axs[0, 0])
    axs[0, 0].set_title("Learned Hamiltonian (H)")
    axs[0, 0].set_xlabel("Theta (rad)")
    axs[0, 0].set_ylabel("Omega (rad/s)")

    im2 = axs[0, 1].contourf(grid_theta, grid_omega, true_H_vals, 20, cmap='viridis')
    fig.colorbar(im2, ax=axs[0, 1])
    axs[0, 1].set_title("Ground Truth Hamiltonian (H)")
    axs[0, 1].set_xlabel("Theta (rad)")

    im3 = axs[0, 2].contourf(grid_theta, grid_omega, np.abs(learned_H_vals - true_H_vals), 20, cmap='Reds')
    fig.colorbar(im3, ax=axs[0, 2])
    axs[0, 2].set_title("Absolute Difference (H)")
    axs[0, 2].set_xlabel("Theta (rad)")

    im4 = axs[1, 0].contourf(grid_theta, grid_omega, learned_R_vals[:, 1, 1].reshape(grid_theta.shape), 20,
                             cmap='cividis')
    fig.colorbar(im4, ax=axs[1, 0])
    axs[1, 0].set_title("Learned R[1,1]")
    axs[1, 0].set_xlabel("Theta (rad)")
    axs[1, 0].set_ylabel("Omega (rad/s)")

    im5 = axs[1, 1].contourf(grid_theta, grid_omega, true_R_vals[:, 1, 1].reshape(grid_theta.shape), 20, cmap='cividis')
    fig.colorbar(im5, ax=axs[1, 1])
    axs[1, 1].set_title("Ground Truth R[1,1]")
    axs[1, 1].set_xlabel("Theta (rad)")

    im6 = axs[1, 2].contourf(grid_theta, grid_omega, learned_G_vals[:, 1].reshape(grid_theta.shape), 20, cmap='Reds')
    fig.colorbar(im6, ax=axs[1, 2])
    axs[1, 2].set_title("Learned G[1,0]")
    axs[1, 2].set_xlabel("Theta (rad)")

    plt.tight_layout()
    plt.show()


# -------------------------------
# Main
# -------------------------------
def main():
    config = load_config(yaml_path)
    # dataset
    X, U, dX = generate_data(config)
    use_traj_dataset = config['training']['preserve_traj']
    if use_traj_dataset:
        seq_len = config['training']['seq_len']
        dataset = TrajectoryStepDataset(X, U, dX, seq_len)
    else:
        dataset = TensorDataset(X, U, dX)
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    # model
    model = pHNN(yaml_path)

    if train:
        model.load_state_dict(torch.load("pendulum_pHNN_weights.pth"))  # loads model config directly
        trained_model = train_pHNN_ode(model, config, dataloader)
        save_model(trained_model, "pendulum_pHNN_weights.pth")
    else:
        model.load_state_dict(torch.load("pendulum_pHNN_weights.pth"))

    print("Learned constant J matrix:")
    print(model.J.detach().numpy())
    evaluate(model, config)
    compare_learned_components(model, config)


if __name__ == "__main__":
    yaml_path = "pendulum_config.yaml"
    train = False  # if continue train model
    main()