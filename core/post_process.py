#post_process.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from prior import WM_Prior2D
import pickle
from line_integral import line_integrals

from matplotlib import rcParams
from pyro.ops.stats import effective_sample_size, gelman_rubin
import arviz as az

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# ----------------------------------------------------------------------
# Matplotlib global style
# ----------------------------------------------------------------------
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'Times']
rcParams['font.size'] = 14
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12

# ----------------------------------------------------------------------
# Thinning helper (based on min ESS)
# ----------------------------------------------------------------------
def thin_by_min_ess(samples_grouped_cpu):
    """
    Thin multi-chain MCMC samples based on the min ESS across KL coefficients.

    Input
    -----
    samples_grouped_cpu : torch.Tensor, shape [C, S, D]
        C = number of chains
        S = samples per chain
        D = number of KL coefficients

    Returns
    -------
    thinned_samples : torch.Tensor, shape [C, S_thin, D]
    min_ess : float
    thinning_factor : int
    """
    C, S, D = samples_grouped_cpu.shape

    ess_per_dim = effective_sample_size(samples_grouped_cpu)  # [D]
    min_ess = ess_per_dim.min().item()

    thinning_factor = max(1, int(round(S / min_ess)))

    print(f"Total samples per chain:   {S}")
    print(f"Min ESS across KL dims:    {min_ess:.2f}")
    print(f"Computed thinning factor:  {thinning_factor}")

    thinned_samples = samples_grouped_cpu[:, ::thinning_factor, :]
    print(f"Thinned samples shape:     {tuple(thinned_samples.shape)}")

    return thinned_samples, min_ess, thinning_factor

# ----------------------------------------------------------------------
# Main post-processing
# ----------------------------------------------------------------------
def run_postprocessing(path, device="cpu", dtype = torch.float32):
    device = torch.device(device)

    # ----------------------------------------------------------
    # 1. Load observation dict (CPU) and move bits we need
    # ----------------------------------------------------------
    with open('./obs/obs_data.pickle', 'rb') as handle:
        obs_dict = pickle.load(handle)

    N_points = obs_dict['N_points']
    N_KL = obs_dict['N_KL']
    mask = obs_dict['mask']
    prior_mean = obs_dict['prior_mean']
    prior_std = obs_dict['prior_std']
    nu = obs_dict['nu']
    tau = obs_dict['tau']
    q = obs_dict['q']
    L = obs_dict['L']
    N_quad = obs_dict['N_quad']
    stations_1 = obs_dict['stations_1']
    stations_2 = obs_dict['stations_2']
    synthetic_field_tensor = obs_dict['synthetic_field_tensor']

     
    # ----------------------------------------------------------
    # 2. Rebuild noise / y_obs (not really needed for plots, but ok)
    # ----------------------------------------------------------
    d_noise = 0.05

    y_true = torch.as_tensor(obs_dict['y_true'], device=device, dtype=dtype)
    noise_vec = torch.as_tensor(obs_dict['noise_vec'], device=device, dtype=dtype)

    sigma = d_noise * torch.norm(y_true) / torch.sqrt(
        torch.as_tensor(y_true.shape[0], device=device, dtype=dtype)
    )
    # s2 = sigma * sigma
    y_obs = y_true + sigma * noise_vec

    # ----------------------------------------------------------
    # 3. Push-forward method
    # ----------------------------------------------------------
    if obs_dict['push_forward_method'] == 'sigmoid':
        k = obs_dict['push_forward_parameter']
        method = lambda x, k=k: torch.sigmoid(k * x)
    else:
        method = lambda x: x

    # ----------------------------------------------------------
    # 4. Load MCMC samples (these were saved on CPU)
    # ----------------------------------------------------------
    with open(path, 'rb') as handle:
        stat_dict = pickle.load(handle)

    # Flat samples: [N_total, N_KL]
    samples_flat_cpu = stat_dict['samples']['x']
    # Grouped samples: [C, S, N_KL] (for diagnostics)
    samples_grouped_cpu = stat_dict['samples'].get('x_grouped', None)

    if samples_grouped_cpu is None:
        # if only flat samples are present, assume a single chain
        samples_grouped_cpu = samples_flat_cpu.unsqueeze(0)

    if not isinstance(samples_grouped_cpu, torch.Tensor):
        samples_grouped_cpu = torch.as_tensor(samples_grouped_cpu)
        
    samples_grouped_cpu = samples_grouped_cpu.to("cpu")
    C, S, D = samples_grouped_cpu.shape
    
    # ----------------------------------------------------------
    # 5. Convergence diagnostics: ESS and R-hat
    # ----------------------------------------------------------
    ess = effective_sample_size(samples_grouped_cpu)   # [N_KL]
    rhat = gelman_rubin(samples_grouped_cpu)           # [N_KL]

    print("##### MCMC diagnostics #####")
    print(f"Number of chains:           {C}")
    print(f"Samples per chain:          {S}")
    print(f"Number of KL coefficients:  {D}")
    print(f"min ESS across KL dims:     {ess.min().item():.2f}")
    print(f"max R-hat across KL dims:   {rhat.max().item():.4f}")
    print("############################\n")

    # Optional: thinning (will usually give factor 1 in your case)
    thinned_grouped, min_ess, thinning_factor = thin_by_min_ess(samples_grouped_cpu)

    # For the rest of the post-processing, we can just use the flat samples
    samples = samples_flat_cpu.to(device=device, dtype=dtype)
    num_samples = samples.shape[0]

    # Mean in coefficient space
    MEAN = torch.mean(samples, dim=0)

    # ----------------------------------------------------------
    # 6. Trace plot of first five KL coefficients (flattened)
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    trace_plot =np.zeros([num_samples,5])
    for i in range(5):
        ax.plot(samples_flat_cpu[:, i].numpy(), lw=0.8)
        trace_plot[:,i] = samples[:, i].numpy()
    ax.set_ylabel('KL coefficient value')
    ax.set_xlabel('Iteration')
    ax.set_title('Trace plot of first five latent KL coefficients')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------
    # 7. Autocorrelation plot for one KL coefficient (multi-chain)
    # ----------------------------------------------------------
    param_index = 0  # e.g. first KL mode
    data_for_arviz = samples_grouped_cpu[:, :, param_index].numpy()  # [C, S]

    inference_data = az.from_dict(posterior={"x_grouped_0": data_for_arviz})
    az.plot_autocorr(inference_data, max_lag = 20)
    plt.tight_layout()
    plt.show()
    
    # ----------------------------------------------------------
    # 8. Build prior and compute posterior mean / std fields
    # ----------------------------------------------------------
    prior = WM_Prior2D(grid_size=N_points, domain_length=L, nu=nu, tau=tau, 
                       device=device, dtype=dtype, push_forward_method=method)

    # pushed_forward: [num_samples, N_points, N_points] on device
    pushed_forward = torch.zeros( num_samples, N_points, N_points, 
                                 device=device, dtype=dtype)

    for i in range(num_samples):
        _, bilevel = prior.WM_priors(samples[i], mask, prior_mean=prior_mean, 
                                     prior_std=prior_std)
        pushed_forward[i, :] = bilevel

    std = torch.std(pushed_forward, dim=0)
    mean_2 = torch.mean(pushed_forward, dim=0)

    # Mean in field space (using mean KL coefficients)
    _, mean_bilevel = prior.WM_priors(MEAN, mask, prior_mean=prior_mean, 
                                      prior_std=prior_std)
    mean_velocity_est = mean_bilevel  # shape (N_points, N_points)
    
    problem = line_integrals(N_pixel=N_points, L=L, N_quad=N_quad, device=device, 
                             dtype=dtype)
    problem.set_stations(stations_1, stations_2)
    velocity_est = mean_velocity_est.reshape(-1)
    y_pred = problem.forward(1.0 / velocity_est)
    # ----------------------------------------------------------
    # 9. Plots: mean and std fields (+ synthetic truth if available)
    # ----------------------------------------------------------
    true_np = synthetic_field_tensor.detach().cpu().numpy()

    f, axes = plt.subplots(1, 3, figsize=(10, 4))

    im0 = axes[0].imshow(true_np, cmap='bwr_r', extent=[0, 1, 0, 1])
    axes[0].set_title('true synthetic field')
    f.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].axis('off')

    mean_vel_np = mean_velocity_est.detach().cpu().numpy().reshape(N_points, N_points)
    im1 = axes[1].imshow(mean_vel_np, cmap='bwr_r', extent=[0, 1, 0, 1])
    axes[1].set_title('mean field')
    f.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].axis('off')
    
    std_np = std.detach().cpu().numpy().reshape(N_points, N_points)
    im2 = axes[2].imshow(std_np, cmap='bwr_r', extent=[0, 1, 0, 1])
    axes[2].set_title('std field')
    f.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].axis('off')
    plt.tight_layout()

    # ------------------------------------------------------
    # 10. Image metrics: SSIM, PSNR, relative error
    # ------------------------------------------------------
    est_np = mean_vel_np

    true_norm = (true_np - true_np.min()) / (true_np.max() - true_np.min())
    est_norm = (est_np - est_np.min()) / (est_np.max() - est_np.min())

    ssim_val = ssim(true_norm, est_norm, data_range=1.0)
    print(f'SSIM: {ssim_val:.4f}')

    psnr_val = psnr(true_norm, est_norm, data_range=1.0)
    print(f"PSNR: {psnr_val:.2f} dB")

    err_velocity_field = (100* torch.norm(synthetic_field_tensor.view(-1) 
                                          - mean_velocity_est.view(-1))/ torch.norm(synthetic_field_tensor.view(-1)))
    print(f'Reconstructed velocity field error: {err_velocity_field:.2f}%')
    
    # ----------------------------------------------------------
    # 11. Travel-time relative error
    # ----------------------------------------------------------
    err_tt = 100 * (torch.norm(y_obs - y_pred) / torch.norm(y_obs))
    print("Travel time Relative error %", float(err_tt))

    # ----------------------------------------------------------
    # 12. Show some posterior samples as fields
    # ----------------------------------------------------------
    f1, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(2):
        for j in range(5):
            idx = i * 5 + j
            _, sample_bilevel = prior.WM_priors(
                samples[idx],
                mask,
                prior_mean=prior_mean,
                prior_std=prior_std,
            )
            sample_np = sample_bilevel.detach().cpu().numpy().reshape(N_points, N_points)

            axes[i, j].imshow(sample_np, extent=[0, 1, 0, 1], cmap='bwr_r')
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = "cpu"  # or "cuda" or "mps"
    dtype = torch.float32
    print("Using device:", device)
    
    path = './stat/stat_paper_code_git.pickle'
    run_postprocessing(path, device=device, dtype = dtype)
    
    
    
    
    
    
    
    
    
    
    
    
    
