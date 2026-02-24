#MAP_estimate.py
import torch
from torch.optim import LBFGS
import numpy as np
import matplotlib.pyplot as plt
from prior import WM_Prior2D
from line_integral import line_integrals
import pickle
import argparse

# Quality metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def run_map(device="cpu", dtype=torch.float64):
    device = torch.device(device)

    # ----------------------------------------------------------
    # 1. Load observation dictionary (stored on CPU)
    # ----------------------------------------------------------
    with open("./obs/obs_data.pickle", "rb") as handle:
        obs_dict = pickle.load(handle)

    N_points = obs_dict["N_points"]
    N_KL = obs_dict["N_KL"]
    prior_mean = obs_dict["prior_mean"]
    prior_std = obs_dict["prior_std"]
    stations_1 = obs_dict["stations_1"]
    stations_2 = obs_dict["stations_2"]
    nu = obs_dict["nu"]
    tau = obs_dict["tau"]
    q = obs_dict["q"]
    L = obs_dict["L"]
    N_quad = obs_dict["N_quad"]
    synthetic_field_tensor= obs_dict["synthetic_field_tensor"]
    mask = obs_dict["mask"]

    # Push-forward method
    if obs_dict["push_forward_method"] == "sigmoid":
        k = obs_dict["push_forward_parameter"]
        method = lambda x, k=k: torch.sigmoid(k * x)
    else:
        method = lambda x: x

    # ----------------------------------------------------------
    # 2. Build prior and forward operator on chosen device
    # ----------------------------------------------------------
    prior = WM_Prior2D(grid_size=N_points, domain_length=L, nu=nu, tau=tau, 
                       device=device, dtype=dtype, push_forward_method=method)

    problem = line_integrals(N_pixel=N_points, L=L, N_quad=N_quad, device=device, 
                             dtype=dtype)
    problem.set_stations(stations_1, stations_2)

    # ----------------------------------------------------------
    # 3. Build data y_obs and noise level sigma
    # ----------------------------------------------------------
    d_noise = 0.05  # relative noise level

    y_true_cpu = obs_dict["y_true"]
    noise_vec_cpu = obs_dict["noise_vec"]

    y_true = torch.as_tensor(y_true_cpu, device=device, dtype=dtype)
    noise_vec = torch.as_tensor(noise_vec_cpu, device=device, dtype=dtype)

    sigma = d_noise * torch.norm(y_true) / torch.sqrt(
        torch.as_tensor(y_true.shape[0], device=device, dtype=dtype))
    s2 = sigma * sigma
    y_obs = y_true + sigma * noise_vec

    # ----------------------------------------------------------
    # 4. MAP optimization with LBFGS on GPU/CPU
    # ----------------------------------------------------------
    # Parameter: KL coefficients
    x = torch.nn.Parameter(torch.randn(N_KL, device=device, dtype=dtype))

    optimizer = LBFGS([x], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()

        # Smooth prior field
        smooth_field, _ = prior.WM_priors(
            x, mask, prior_mean=prior_mean, prior_std=prior_std)
        velocity = smooth_field.reshape(-1)

        # Forward model uses NF / velocity as slowness
        y_pred = problem.forward(1.0 / velocity)

        # L2 data misfit + Gaussian prior on x
        loss = torch.sum((y_pred - y_obs) ** 2) + s2 * torch.sum(x**2)
        loss.backward()

        print("Loss:", float(loss))
        return loss

    optimizer.step(closure)

    # ----------------------------------------------------------
    # 5. Recompute velocity field (here using bi-level field)
    # ----------------------------------------------------------
    _, bilevel_field = prior.WM_priors(
        x, mask, prior_mean=prior_mean, prior_std=prior_std
    )
    velocity_est = bilevel_field.reshape(-1)

    y_pred = problem.forward(1.0 / velocity_est)

    # ----------------------------------------------------------
    # 6. Plots: move everything to CPU
    # ----------------------------------------------------------
    y_pred_cpu = y_pred.detach().cpu().numpy()
    y_obs_cpu = y_obs.detach().cpu().numpy()

    plt.figure()
    plt.plot(y_pred_cpu[:1000], label="simulated")
    plt.plot(y_obs_cpu[:1000], "r", label="real")
    plt.legend()
    plt.title("travel_time")
    plt.show()

    # ----------------------------------------------------------
    # 7. Synthetic vs real cases
    # ----------------------------------------------------------
    vel_map_cpu = velocity_est.detach().cpu().numpy().reshape(N_points, N_points)

    plt.figure()
    plt.imshow(vel_map_cpu, extent=[0, L, 0, L], cmap="bwr_r")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.axis("off")
    plt.title("MAP estimation field")


    true_np = synthetic_field_tensor.detach().cpu().numpy()
    plt.figure()
    plt.imshow(true_np, extent=[0, L, 0, L], cmap="bwr_r")
    plt.title("synthetic field")
    plt.colorbar()
    plt.axis("off")

    est_np = vel_map_cpu

    # Normalize both to [0, 1]
    true_norm = (true_np - true_np.min()) / (true_np.max() - true_np.min())
    est_norm = (est_np - est_np.min()) / (est_np.max() - est_np.min())

    ssim_val = ssim(true_norm, est_norm, data_range=1.0)
    print(f"SSIM: {ssim_val:.4f}")

    psnr_val = psnr(true_norm, est_norm, data_range=1.0)
    print(f"PSNR: {psnr_val:.2f} dB")

    err_velocity_field = 100 * torch.norm(synthetic_field_tensor.view(-1)
                                          - velocity_est.cpu().view(-1))/ torch.norm(synthetic_field_tensor.view(-1))
    
    print(f"Reconstructed velocity field error: {err_velocity_field:.2f}%")
    
    # ----------------------------------------------------------
    # 8. Travel-time relative error
    # ----------------------------------------------------------
    err_tt = 100 * (torch.norm(y_obs - y_pred) / torch.norm(y_obs))
    print("Travel time Relative error %", float(err_tt))

    plt.show()

    return x, velocity_est, y_pred


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")  # "cpu" or "cuda" or "mps"
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args()

    device = args.device
    dtype = getattr(torch, args.dtype)
    
    run_map(device = device, dtype = dtype)
