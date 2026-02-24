#line_integral.py
import torch
import matplotlib.pyplot as plt
import pickle
import time
from time import perf_counter

def sync_for(device):
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.synchronize
    if device.type == "mps" and torch.backends.mps.is_available():
        return torch.mps.synchronize
    return lambda: None

def bench_forward(N_pixel=512, N_quad=512, device="cpu", dtype=torch.float32):
    problem = line_integrals(N_pixel=N_pixel, L=1.0, N_quad=N_quad,
                             device=device, dtype=dtype)
    s1, s2 = problem.generate_station_pairs(M=14, mode="even")
    problem.set_stations(s1, s2)

    field = torch.ones(N_pixel, N_pixel, device=device, dtype=dtype)
    sync = sync_for(device)

    # warmup
    for _ in range(3):
        _ = problem.forward(field.view(-1))
        sync()

    # timed
    t0 = perf_counter()
    out = problem.forward(field.view(-1))
    sync()
    t1 = perf_counter()

    return out, t1 - t0


class line_integrals:  
    def __init__(self, N_pixel, L=1.0, N_quad=256, device=None, dtype=torch.float32):
        
        """
        Line-integral forward operator on a uniform N_pixel x N_pixel grid.
        """
        
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.dtype = dtype
        self.N_pixel, self.N_quad, self.L = N_pixel, N_quad, L
        
        # keep coords on CPU (not used later), or put device=self.device if you prefer
        x = torch.linspace(0.0, self.L, self.N_pixel, dtype=self.dtype)
        y = torch.linspace(0.0, self.L, self.N_pixel, dtype=self.dtype)
        im_x, im_y = torch.meshgrid(x, y, indexing='ij')
        self.im_x, self.im_y = im_x.reshape(-1), im_y.reshape(-1)

        self.indices = None
        self.dx = None

    def generate_station_pairs(self, M: int, mode: str = 'even'):
        """
        Generate station pairs.
        M: number of stations per side (total M^2 stations).
        mode: 'random' or 'even'.
        """
        if mode == 'random':
            s_x = self.L * torch.rand(M**2, dtype=self.dtype)
            s_y = self.L * torch.rand(M**2, dtype=self.dtype)
        elif mode == 'even':
            s_axis = torch.linspace(0.0, self.L, M, dtype=self.dtype)
            grid_x, grid_y = torch.meshgrid(s_axis, s_axis, indexing='ij')
            s_x = grid_x.reshape(-1)
            s_y = grid_y.reshape(-1)
        else:
            raise ValueError("mode must be either 'random' or 'even'")

        # Build all distinct pairs (i, j), i != j
        IDX = torch.arange(s_x.shape[0], dtype=torch.int32)
        EYE = torch.ones(s_x.shape[0], dtype=torch.int32)

        pairs = torch.cat(
            [
                torch.kron(IDX, EYE).reshape(-1, 1),
                torch.kron(EYE, IDX).reshape(-1, 1),
            ],
            dim=1,
        )
        mask_pair = pairs[:, 0] != pairs[:, 1]
        pairs = pairs[mask_pair]

        p1x = s_x[pairs[:, 0].long()]
        p1y = s_y[pairs[:, 0].long()]
        p2x = s_x[pairs[:, 1].long()]
        p2y = s_y[pairs[:, 1].long()]

        s1 = (p1x, p1y)
        s2 = (p2x, p2y)

        return s1, s2

    def set_stations(self, s1, s2):
        """
        Precompute quadrature point indices and step lengths for given station pairs.
        """
        p1x, p1y = s1
        p2x, p2y = s2

        vx = p2x - p1x
        vy = p2y - p1y
        normalizer = torch.sqrt(vx**2 + vy**2)

        vx = vx / normalizer
        vy = vy / normalizer
        dx = normalizer / self.N_quad  # (n_pairs,)

        # Quadrature points along each ray
        quad_idx = torch.arange(self.N_quad, dtype=self.dtype).view(1, -1)
        quad_points = quad_idx * dx.view(-1, 1)  # (n_pairs, N_quad)

        quad_px = p1x.view(-1, 1) + quad_points * vx.view(-1, 1)
        quad_py = p1y.view(-1, 1) + quad_points * vy.view(-1, 1)

        # Map coordinates to pixel indices
        px_idx = torch.clamp((quad_px * self.N_pixel / self.L).long(), 0, self.N_pixel - 1)
        py_idx = torch.clamp((quad_py * self.N_pixel / self.L).long(), 0, self.N_pixel - 1)

        self.indices = (px_idx * self.N_pixel + py_idx).long().contiguous()
        self.dx = dx.to(self.dtype)  # stays float
        
        # move once if using GPU/MPS
        if self.device.type != "cpu":
            self.indices = self.indices.to(self.device, non_blocking=False)
            self.dx = self.dx.to(self.device, non_blocking=False)
            
            
    def forward(self, im: torch.Tensor) -> torch.Tensor:
        if self.indices is None or self.dx is None:
            raise RuntimeError("You must call set_stations(...) before forward().")
    
        if im.device != self.device:
            im = im.to(self.device)
    
        # im[self.indices]: (n_pairs, N_quad)
        return torch.sum(im[self.indices], dim=1) * self.dx

# ----------------------------------------------------------------------
# Test helpers 
# ----------------------------------------------------------------------

def test_line_integral(device="cpu", dtype=torch.float32):
    # Parameters
    N_pixel = 256
    N_quad = 256
    L = 1.0
    M = 14               # number of stations along one side
    station_mode = 'even'

    problem = line_integrals(N_pixel=N_pixel, L=L, N_quad=N_quad, 
                             device=device, dtype=dtype)

    s1, s2 = problem.generate_station_pairs(M=M, mode=station_mode)
    problem.set_stations(s1, s2)

    field = torch.ones(N_pixel, N_pixel, device=device, dtype=dtype)
    out = problem.forward(field.reshape(-1))

    print("Travel times: ", out)

    plt.figure()
    plt.plot(out.detach().cpu().numpy())
    plt.title("Line-integral of ones field")
    plt.show()

    s1x, s1y = s1
    s2x, s2y = s2
    distance = torch.sqrt((s2x - s1x)**2 + (s2y - s1y)**2)
    print("Station pairs distance: ", distance)

    err = 100 * torch.norm(distance - out) / torch.norm(distance)
    print("Forward error (%): ", err.item())


def test_line_integral_real_stations(N_pixel = 256, N_quad = 256, device="cpu", dtype=torch.float64):
    with open("./real_data/normalized_real_data_in_XY.pickle", "rb") as handle:
        obs_dict = pickle.load(handle)

    s1 = [obs_dict["s_xn1"], obs_dict["s_yn1"]]  # normalized station 1
    s2 = [obs_dict["s_xn2"], obs_dict["s_yn2"]]  # normalized station 2

    # Convert stations to tensors on chosen device
    s1x = torch.tensor(s1[0], dtype=dtype)
    s1y = torch.tensor(s1[1], dtype=dtype)
    s2x = torch.tensor(s2[0], dtype=dtype)
    s2y = torch.tensor(s2[1], dtype=dtype)
    s1 = (s1x, s1y)
    s2 = (s2x, s2y)

    L = obs_dict["L"]

    
    

    problem = line_integrals(N_pixel=N_pixel, L=L, N_quad=N_quad, device=device, 
                             dtype = dtype)
    problem.set_stations(s1, s2)

    field = torch.ones(N_pixel, N_pixel, device=device, dtype=dtype)
    out = problem.forward(field.reshape(-1)).cpu()

    print(out)

    plt.figure()
    plt.plot(out[:5000].detach().cpu().numpy())
    plt.title("Line-integral (real stations)")
    plt.show()

    distance = torch.sqrt((s2x - s1x)**2 + (s2y - s1y)**2)
    print("Station pairs distance: ", distance)

    err = 100 * torch.norm(distance - out) / torch.norm(distance)
    print("Forward error (%): ", err.item())


if __name__ == "__main__":
    # Choose device: "cpu" or "cuda"
    device = "mps"  # or "cuda" or "mps"
    dtype = torch.float32
    print("Using device:", device)
    
    N_pixel = 512
    N_quad = 512
    # test_line_integral(device=dev, dtype=torch.float32)
    test_line_integral_real_stations(N_pixel, N_quad, device = device, dtype=torch.float32)
    
    out_cpu, t_cpu = bench_forward(device="cpu", dtype=torch.float32)
    out_mps, t_mps = bench_forward(device="mps", dtype=torch.float32)
    print(f"CPU: {t_cpu:.4f}s   MPS: {t_mps:.4f}s   speedup x{t_cpu/max(t_mps,1e-9):.2f}")
        