#prior.py
import torch
import matplotlib.pyplot as plt


class WM_Prior2D:
    """
    Sample 2D WM prior via FFT.

    """
    def __init__(self, grid_size: int, domain_length: float = 1.0, 
                 nu: float = 1.5, tau: float = 2.0, q: float = 1.0,
                 device = None, dtype: torch.dtype = torch.float32,
                 push_forward_method = None):
        
        if device is None:
            self.device = torch.device("cpu")
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device)
        
        self.dtype = torch.float32 if (self.device.type == "mps" and dtype == torch.float64) else dtype
        

        self.N = grid_size
        self.L = domain_length
        self.nu = nu
        self.tau = tau
        self.q = q
        self.dtype = dtype

        dx = self.L / self.N

        # Fourier wavenumbers (2π is not included here)
        k1 = torch.fft.fftfreq(self.N, dx, dtype=self.dtype) * 2 * self.L
        kx, ky = torch.meshgrid(k1, k1, indexing="ij")
        k_squared = kx**2 + ky**2
        
        # Move to target device once
        self.k_squared = k_squared.to(self.device)

        # Power spectrum S(k) ∝ (tau^2 + ||k||^2)^(-(nu + d/2)), with d=2 → exponent = nu + 1
        alpha = self.nu + 1.0
        spectral_density = torch.pow(self.tau**2 + self.k_squared, -alpha)
        
        # Normalize to avoid exploding magnitudes
        spectral_density = spectral_density / spectral_density.norm()

        # Store sqrt of spectral density
        self.sqrt_spectral_density = torch.sqrt(spectral_density)

        # Precompute useful scalars on correct device/dtype
        self.sqrt_q = torch.sqrt(torch.as_tensor(self.q, device=self.device, dtype=self.dtype))
        self.sqrt2 = torch.sqrt(torch.as_tensor(2.0, device=self.device, dtype=self.dtype))

        # push-forward
        self.push_forward_method = (push_forward_method if push_forward_method
                                    is not None else lambda x: x)
       
        self.last_smooth_field = None
        self.last_bilevel_field = None
        
    def get_sqrt_spectral_density(self) -> torch.Tensor:
        """Return the stored sqrt of the spectral density."""
        return self.sqrt_spectral_density

    def WM_priors(self, psi: torch.Tensor, mask: torch.Tensor,
                  prior_mean: float = 0.0, prior_std: float = 1.0):

        mask = mask.to(self.device)
        psi_hat = torch.zeros((self.N, self.N), device=self.device, dtype=self.dtype)
        
        psi_hat[mask] = psi.to(self.device, dtype=self.dtype)

        # Apply spectral filter
        filtered_hat = self.sqrt_spectral_density * psi_hat

        # --- FFT path: CPU fallback for MPS ---
        if self.device.type == "mps":
            filtered_hat_cpu = filtered_hat.to("cpu")
            sqrt_q_cpu = self.sqrt_q.to("cpu")
            complex_field_cpu = torch.fft.ifft2(sqrt_q_cpu * filtered_hat_cpu)

            # Convert to real before moving back
            real_field_cpu =  self.sqrt2.to("cpu")*(complex_field_cpu.real + complex_field_cpu.imag)
            # real_field_cpu =  (complex_field_cpu.real).to("cpu")
            real_field = real_field_cpu.to(self.device)
        else:
            complex_field = torch.fft.ifft2(self.sqrt_q * filtered_hat)
            real_field =  self.sqrt2*(complex_field.real + complex_field.imag)
            # real_field =  (complex_field.real)
        # ---------------------------------------------------

        # Mean/std as tensors
        mean_t = torch.as_tensor(prior_mean, device=self.device, dtype=self.dtype)
        std_t = torch.as_tensor(prior_std, device=self.device, dtype=self.dtype)

        # Smooth transform
        smooth_field = mean_t + std_t * self.push_forward_method(real_field)

        # Bi-level field
        step = (real_field > 0).to(self.dtype)
        bilevel_field = mean_t + std_t * step

        self.last_smooth_field = smooth_field
        self.last_bilevel_field = bilevel_field
        return smooth_field, bilevel_field


class SpectralEnergyMask2D:
    """
    Compute a boolean mask that keeps the modes containing a given fraction of
    total spectral energy.
    """

    def __init__(self, sqrt_spectral_density: torch.Tensor):
        """
            sqrt_spectral_density: 2D tensor of sqrt of spectral density, shape (N, N).
        """
        assert sqrt_spectral_density.ndim == 2, "Spectral density must be 2D."
        self.sqrt_spectral_density = sqrt_spectral_density
        self.N = sqrt_spectral_density.shape[0]
        self.mask = torch.ones_like(sqrt_spectral_density, dtype=torch.bool)
        self.N_KL = self.N * self.N

    def compute_mask(self, energy_fraction: float = 0.95):
        """
        Compute a mask that retains the given fraction of total spectral energy.

        """
        # Spectral density S(k) = (sqrt_spectral_density)^2
        S = self.sqrt_spectral_density**2
        S_flat = S.flatten()

        # Sort energies in descending order
        S_sorted, indices = torch.sort(S_flat, descending=True)

        # Cumulative energy ratio
        cumulative_energy = torch.cumsum(S_sorted, dim=0)
        total_energy = cumulative_energy[-1]
        ratio = cumulative_energy / total_energy

        # Smallest number of modes to reach desired energy fraction
        idx = torch.searchsorted(ratio, energy_fraction)
        num_selected = int(idx.item() + 1)

        # Build boolean mask
        mask_flat = torch.zeros_like(S_flat, dtype=torch.bool)
        mask_flat[indices[:num_selected]] = True

        self.mask = mask_flat.view(self.N, self.N)
        self.N_KL = num_selected

        print(f"[SpectralEnergyMask2D] Selected {num_selected} modes "
              f"to retain ≥ {energy_fraction * 100:.1f}% energy.")

        return self.mask, self.N_KL

def plot_2D():
    N_points = 512
    L = 1
    nu = 2
    tau = 10
    
    k = 500000
    method = lambda x, k=k: torch.sigmoid(k * x)
    
   
   
    dtype = torch.float32
    prior = WM_Prior2D(grid_size = N_points, domain_length = L, nu = nu, 
                       tau = tau, dtype = dtype,
                       push_forward_method = method)
        
    
    sqrt_sd = prior.get_sqrt_spectral_density()
    
    masker = SpectralEnergyMask2D(sqrt_sd)
    mask, N_KL = masker.compute_mask(energy_fraction = 1)

    f,axes = plt.subplots(2,2)
    for i in range(2):
        for j in range(2):
            psi = torch.randn(N_KL, dtype=torch.float32)
            psi.requires_grad_(True)
            
            smooth, bilevel  = prior.WM_priors(psi,mask,  prior_mean =1 , prior_std = 1)
            
            loss = torch.norm(smooth)
            loss.backward()
            # print(psi.grad)
            
            
            fig = axes[i,j].imshow(bilevel.detach().cpu().numpy(), cmap='bwr_r')
            axes[i,j].set_axis_off()
            plt.colorbar(fig, ax=axes[i,j])
            axes[i,j].set_title(f'prior sample {i+j+1}')
   
    plt.show()



        
if __name__ == '__main__':
    plot_2D()
    
    