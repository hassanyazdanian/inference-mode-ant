#save_signal.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from prior import WM_Prior2D, SpectralEnergyMask2D
from line_integral import line_integrals
import pickle
import argparse



def synthetic_phantom(N_points=310, N_quad=256, L=1,dtype=torch.float64, k=5e4, 
                      nu=2, tau=15, q=1, prior_mean=2, prior_std=1, device = "cpu",
                      M=14, station_mode ='even'):
    
    device = torch.device(device)
    # -----------------------------------------------------------
    # 1. Load synthetic field and pad
    # -----------------------------------------------------------
    synthetic_field = np.load("./make_synthetic_field/synthetic_field.npy")

    pad_height = N_points - synthetic_field.shape[0]
    pad_width = N_points - synthetic_field.shape[1]

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    synthetic_field = np.pad(synthetic_field,((pad_top, pad_bottom), 
                                              (pad_left, pad_right)), mode="edge")

    synthetic_field_tensor = torch.tensor(synthetic_field, dtype=dtype)

    # -----------------------------------------------------------
    # 2. Prior + mask on chosen device
    # -----------------------------------------------------------
    method_name = "sigmoid"
    method = lambda x, k=k: torch.sigmoid(k * x)

    prior = WM_Prior2D(grid_size=N_points, domain_length=L, nu=nu, tau=tau, 
                       dtype=dtype, push_forward_method=method, device = device)

    sqrt_sd = prior.get_sqrt_spectral_density()
    masker = SpectralEnergyMask2D(sqrt_sd)
    mask, N_KL = masker.compute_mask(energy_fraction = 0.99)
    print("N_KL =", N_KL)

    # Sample coefficients on device
    p_true = torch.randn(N_KL, dtype=dtype)

    out_smooth, out_bilevel = prior.WM_priors(p_true, mask, 
                                              prior_mean=prior_mean, prior_std=prior_std)

    plt.figure()
    plt.imshow(np.flipud(out_smooth.detach().numpy()), 
               extent=[0, L, 0, L], cmap="bwr_r")
    plt.colorbar()
    plt.title("prior sample")

    # -----------------------------------------------------------
    # 3. Build forward problem and stations
    # -----------------------------------------------------------
    problem = line_integrals(N_points, L = L, N_quad=N_quad, dtype = dtype)
    
    s1, s2 = problem.generate_station_pairs(M=M, mode=station_mode)
    s1x, s1y = s1
    s2x, s2y = s2
    s_x = (s1x, s2x)
    s_y = (s1y, s2y)
    problem.set_stations(s1, s2)

    # -----------------------------------------------------------
    # 4. Forward data + noise
    # -----------------------------------------------------------
    slowness = (1.0 / synthetic_field_tensor).reshape(-1)
    y_true = problem.forward(slowness)

    noise_vec = torch.randn_like(y_true)
    noise_vec = noise_vec / torch.norm(noise_vec)

    # -----------------------------------------------------------
    # 6. Save obs dict (CPU for portability)
    # -----------------------------------------------------------
    obs_dict = { "N_points": N_points, "N_KL": N_KL, "mask": mask,
                "synthetic_field_tensor": synthetic_field_tensor,
                "prior_mean": prior_mean, "prior_std": prior_std, 
                "push_forward_method": method_name, "push_forward_parameter": k,
                "stations_1": (s_x[0], s_y[0]),"stations_2": (s_x[1], s_y[1]),
                "y_true": y_true, "noise_vec": noise_vec, "nu": nu, "tau": tau,
                "q": q, "L": L, "N_quad": N_quad}

    with open("./obs/obs_data.pickle", "wb") as handle:
        pickle.dump(obs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # -----------------------------------------------------------
    # 7. Plot phantom + stations
    # -----------------------------------------------------------
    f, ax = plt.subplots()
    im = ax.imshow( synthetic_field_tensor.numpy(), extent=[0, L, 0, L], 
                   cmap="bwr_r")
    f.colorbar(im, ax=ax)
    ax.scatter(torch.stack(s_x), torch.stack(s_y), label="Stations", 
               linewidths=0.01, color="white", marker=".")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="lower left")
    plt.show()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")  # "cpu" or "cuda" or "mps"
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args()

    device = args.device
    dtype = getattr(torch, args.dtype)
    
    
    synthetic_phantom(N_points = 310, N_quad = 450, L = 1, dtype = dtype, k= 5e4, 
                      nu = 2, tau = 15, q=1, prior_mean=2, prior_std=1, M=14,
                      station_mode="even")


    
    
    
    
