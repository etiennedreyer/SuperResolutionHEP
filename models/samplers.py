import torch
from tqdm import tqdm
import numpy as np
from scipy import integrate

@torch.no_grad()
def edm_sampler(
    net,
    batch,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    def round_sigma(sigma):
        return torch.as_tensor(sigma)

    e_proxy = batch["e_proxy"]

    seq = []
    latents = torch.randn(
        (e_proxy.shape[0], e_proxy.shape[1], 1), device=e_proxy.device
    )

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(
        num_steps, dtype=torch.float64, device=latents.device
    )
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    # Main sampling loop.
    x_next = latents * t_steps[0]
    for i, (t_cur, t_next) in tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])), total=num_steps
    ):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1)
            if S_min <= t_cur <= S_max
            else 0
        )
        t_hat = round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(
            x_cur
        )
        # Euler step.
        sigma = t_hat.unsqueeze(0).repeat(e_proxy.shape[0])
        denoised = net(batch, x_hat, sigma)

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            # sigma = t_next.unsqueeze(0).repeat(gp.batch_size)
            sigma = t_next.unsqueeze(0).repeat(e_proxy.shape[0])
            denoised = net(batch, x_next, sigma)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next

@torch.no_grad()
def dpm2_sampler(
    net,
    g,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    def round_sigma(sigma):
        return torch.as_tensor(sigma)

    def to_d(x, sigma, denoised):
        """Converts a denoiser output to a Karras ODE derivative."""
        return (x - denoised) / sigma

    seq = []
    gp = net.infer_graph(g)
    latents = torch.randn(
        (gp.num_nodes("fastsim_particles"), 3), device=gp.device
    )  # .to(torch.float64)
    gp.nodes["fastsim_particles"].data["latents"] = latents
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    step_indices = torch.arange(
        num_steps, dtype=torch.float64, device=latents.device
    )
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    x_next = latents * t_steps[0]
    for i, (t_cur, t_next) in tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])), total=num_steps
    ):
        x_cur = x_next
        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1)
            if S_min <= t_cur <= S_max
            else 0
        )

        t_hat = round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(
            x_cur
        )
        gp.nodes["fastsim_particles"].data["pt_eta_phi_corrupted"] = x_hat
        # Euler step.
        sigma = t_hat.unsqueeze(0).repeat(gp.batch_size)
        gp = net(gp, sigma)
        denoised = gp.nodes["fastsim_particles"].data[
            "pt_eta_phi_pred"
        ]  # .to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat

        if i == num_steps - 1:
            # Euler method
            x_next = x_hat + (t_next - t_hat) * d_cur
            gp.nodes["fastsim_particles"].data["pt_eta_phi_corrupted"] = x_next
        else:
            # DPM-Solver-2
            t_mid = t_hat.log().lerp(t_next.log(), 0.5).exp()
            dt_1 = t_mid - t_hat
            dt_2 = t_next - t_hat
            x_2 = x_hat + d_cur * dt_1
            gp.nodes["fastsim_particles"].data["pt_eta_phi_corrupted"] = x_2
            sigma_mid = t_mid.unsqueeze(0).repeat(gp.batch_size)

            g = net(g, sigma_mid)
            denoised_2 = gp.nodes["fastsim_particles"].data["pt_eta_phi_pred"]
            d_2 = (x_2 - denoised_2) / t_mid
            # d_2 = to_d(x_2, sigma_mid, denoised_2)
            x_next = x_hat + d_2 * dt_2
    return gp, seq



def linear_multistep_coeff(self, order, t, i, j):
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    def fn(tau):
        prod = 1.0
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]


@torch.no_grad()
def sample_lms(
    self, g, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7, order=4
):
    def round_sigma(sigma):
        return torch.as_tensor(sigma)

    def to_d(x, sigma, denoised):
        """Converts a denoiser output to a Karras ODE derivative."""
        return (x - denoised) / sigma

    seq = []
    gp = self.net.infer_graph(g)
    latents = torch.randn(
        (gp.num_nodes("fastsim_particles"), 3), device=gp.device
    )  # .to(torch.float64)
    gp.nodes["fastsim_particles"].data["latents"] = latents
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, self.net.sigma_min)
    sigma_max = min(sigma_max, self.net.sigma_max)

    step_indices = torch.arange(
        num_steps, dtype=torch.float64, device=latents.device
    )
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    x_next = latents * t_steps[0]
    t_steps_cpu = t_steps.detach().cpu().numpy()
    ds = []
    for i in tqdm(range(len(t_steps) - 1)):
        x_cur = x_next
        gp.nodes["fastsim_particles"].data["pt_eta_phi_corrupted"] = x_cur
        sigma = t_steps[i].unsqueeze(0).repeat(gp.batch_size)
        denoised = (
            self.net(gp, sigma).nodes["fastsim_particles"].data["pt_eta_phi_pred"]
        )
        # d = to_d(x_cur, sigma, denoised)
        # print(x_cur.shape, denoised.shape, sigma.shape)
        d = (x_cur - denoised) / t_steps[i]
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        cur_order = min(i + 1, order, len(t_steps) - i - 1)
        coeffs = [
            self.linear_multistep_coeff(cur_order, t_steps_cpu, i, j)
            for j in range(cur_order)
        ]
        x = x_cur + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
        x_next = x
    return gp, seq  # , t_steps
