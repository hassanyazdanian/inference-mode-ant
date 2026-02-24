#sampling.py
import torch
from pyro.infer.mcmc import MCMC, NUTS
from prior import WM_Prior2D
from line_integral import line_integrals
import pickle
import argparse
from pyro.ops.stats import effective_sample_size, gelman_rubin
import time

def run_nuts(saving_path, device="cpu", dtype=torch.float32, num_chains = 4,
             num_samples = 400, warmup_steps = 250):
    device = torch.device(device)

    # ----------------------------------------------------------
    # 1. Load observation dictionary
    # ----------------------------------------------------------
    with open("./obs/obs_data.pickle", "rb") as handle:
        obs_dict = pickle.load(handle)

    N_points   = obs_dict["N_points"]
    N_KL       = obs_dict["N_KL"]
    prior_mean = obs_dict["prior_mean"]
    prior_std  = obs_dict["prior_std"]
    stations_1 = obs_dict["stations_1"]
    stations_2 = obs_dict["stations_2"]
    nu         = obs_dict["nu"]
    tau        = obs_dict["tau"]
    q          = obs_dict["q"]
    L          = obs_dict["L"]
    N_quad     = obs_dict["N_quad"]
    mask       = obs_dict["mask"]


    # Push-forward method
    if obs_dict["push_forward_method"] == "sigmoid":
        k = obs_dict["push_forward_parameter"]
        method = lambda x, k=k: torch.sigmoid(k * x)
    else:
        method = lambda x: x

    # ----------------------------------------------------------
    # 2. Build prior and forward operator on device
    # ----------------------------------------------------------
    prior = WM_Prior2D(grid_size=N_points, domain_length=L, nu=nu, tau=tau, 
                       device=device, dtype=dtype, push_forward_method=method)

    problem = line_integrals(N_pixel=N_points, L=L, N_quad=N_quad, device=device, 
                             dtype=dtype)
    problem.set_stations(stations_1, stations_2)

    # ----------------------------------------------------------
    # 3. Build y_obs and noise variance s2 on device
    # ----------------------------------------------------------
    d_noise = 0.05

    y_true_cpu = obs_dict["y_true"]
    noise_vec_cpu = obs_dict["noise_vec"]

    y_true = torch.as_tensor(y_true_cpu, device=device, dtype=dtype)
    noise_vec = torch.as_tensor(noise_vec_cpu, device=device, dtype=dtype)

    sigma = d_noise * torch.norm(y_true) / torch.sqrt(
        torch.as_tensor(y_true.shape[0], device=device, dtype=dtype)
    )
    s2 = sigma * sigma
    y_obs = y_true + sigma * noise_vec

    # ----------------------------------------------------------
    # 4. Potential function (negative log-posterior)
    # ----------------------------------------------------------
    def pyro_NLP(params):
        """
        params: dict with key 'x'
        Returns: scalar potential (negative log-posterior)
        """
        x = params["x"]  # (N_KL,) on chosen device

        # Smooth prior field
        smooth_field, _ = prior.WM_priors(
            x, mask, prior_mean=prior_mean, prior_std=prior_std
        )
        velocity = smooth_field.reshape(-1)

        # Forward model (NF/velocity is slowness)
        y_pred = problem.forward(1.0 / velocity)

        # Data misfit + Gaussian prior on x
        misfit = torch.sum((y_pred - y_obs) ** 2)/s2
        prior_term = torch.sum(x**2)

        return misfit + prior_term

    # ----------------------------------------------------------
    # 5. NUTS + MCMC on device
    # ----------------------------------------------------------
    initial_params = {"x": torch.zeros(N_KL, device=device, dtype=dtype)}
    
    
    all_samples = []
    chain_times = []
    
    for chain in range(num_chains):
        print(f"Running chain {chain+1}/{num_chains}...")
        
        start_time = time.time()
        
        # fresh kernel per chain
        nuts_kernel = NUTS(potential_fn=pyro_NLP)
        
        mcmc = MCMC(
            kernel=nuts_kernel,
            num_chains=1,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            initial_params=initial_params,
        )
        mcmc.run()
        # mcmc.summary()
        
        end_time = time.time()
        chain_runtime = end_time - start_time
        chain_times.append(chain_runtime)
        print(f"Chain {chain+1} runtime: {chain_runtime/60:.2f} minutes")

        
        chain_samples = mcmc.get_samples()["x"]  # [num_samples, N_KL]
        all_samples.append(chain_samples.unsqueeze(0)) #[1, num_samples, N_KL]
    
    
    total_runtime = sum(chain_times)
    print(f"\nTotal runtime for all chains: {total_runtime/60:.2f} minutes")
    
    
    samples_grouped = torch.cat(all_samples, dim=0)  # Stack: [C, S, N_KL]
    
    # Move to CPU
    samples_grouped_cpu = samples_grouped.detach().cpu()  # [C, S, N_KL]
    # ----------------------------------------------------------
    # 6. Diagnostics (ESS, R-hat)
    # ----------------------------------------------------------
    # shape [N_KL]
    ess  = effective_sample_size(samples_grouped_cpu)
    rhat = gelman_rubin(samples_grouped_cpu)
  
    print(f"min ESS across x:  {ess.min().item():.1f}")
    print(f"max R-hat across x:{rhat.max().item():.4f}")
    
    # ----------------------------------------------------------
    # 7. Collect samples and move them to CPU for saving
    # ----------------------------------------------------------
    # Move to CPU
    samples_flat_cpu    = samples_grouped_cpu.reshape(-1, N_KL) # [C*S, N_KL]
    
    samples_cpu = {
        "x": samples_flat_cpu,              # flattened samples for post-processing
        "x_grouped": samples_grouped_cpu,   # per-chain samples for diagnostics
    }
    
    stat_dict = {"samples": samples_cpu}
    
    out_file = saving_path
    with open(out_file, "wb") as handle:
        pickle.dump(stat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Saved samples to {out_file}")
    return samples_cpu

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")  # "cpu" or "cuda" or "mps"
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args()

    device = args.device
    dtype = getattr(torch, args.dtype)
    
    print("Using device:", device)
    
    saving_path = './stat/stat_paper_code_git.pickle'
    num_chains = 4
    num_samples = 400
    warmup_steps = 250 
    samples = run_nuts(saving_path, device=device, dtype = dtype,
                       num_chains = num_chains, num_samples = num_samples, 
                       warmup_steps = warmup_steps)
