import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange

from tqdm.auto import tqdm

from dit import DiT


def normalize_to_neg1_1(x):
    return x * 2 - 1

def unnormalize_to_0_1(x):
    return (x + 1) * 0.5

class RectifiedFlow(nn.Module):
    def __init__(
        self,
        net: DiT,
        device="cuda",
        channels=3,
        image_size=32,
        num_classes=10,
        logit_normal_sampling_t=True,
    ):
        super().__init__()
        self.net = net
        self.device = device
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = num_classes is not None
        self.logit_normal_sampling_t = logit_normal_sampling_t

    def forward(self, x, c=None, encoder_features=None):
        """
        Forward pass for training.
        
        Args:
            x: Input images (B, C, H, W) in [0, 1] range
            c: Class labels (B,)
            encoder_features: Optional list of encoder features for REPA alignment
        
        Returns:
            If encoder_features is None: denoising loss
            If encoder_features provided: (denoising_loss, alignment_loss)
        """
        if self.logit_normal_sampling_t:
            t = torch.randn((x.shape[0],), device=self.device).sigmoid()
        else:
            t = torch.rand((x.shape[0],), device=self.device)
        
        t_ = rearrange(t, "b -> b 1 1 1")
        z = torch.randn_like(x)
        x = normalize_to_neg1_1(x)
        z_t = (1 - t_) * x + t_ * z
        target = z - x
        
        # Check if we need to compute alignment loss
        if encoder_features is not None:
            v_t, zs_model, _ = self.net(z_t, t, c, return_features=True)
            denoising_loss = F.mse_loss(target, v_t)
            
            # Compute alignment loss
            alignment_loss = 0.0
            if zs_model is not None:
                for z_model, z_enc in zip(zs_model, encoder_features):
                    alignment_loss = alignment_loss + self._compute_alignment_loss(z_model, z_enc)
                alignment_loss = alignment_loss / len(encoder_features)
            
            return denoising_loss, alignment_loss
        else:
            v_t = self.net(z_t, t, c)
            return F.mse_loss(target, v_t)
    
    def _compute_alignment_loss(self, z_model, z_encoder):
        """Compute cosine similarity alignment loss."""
        # Normalize both
        z_model = F.normalize(z_model, dim=-1)
        z_encoder = F.normalize(z_encoder, dim=-1)
        
        # Handle sequence length mismatch
        if z_model.shape[1] != z_encoder.shape[1]:
            B, T_m, D_m = z_model.shape
            B, T_e, D_e = z_encoder.shape
            
            H_m = int(T_m ** 0.5)
            H_e = int(T_e ** 0.5)
            
            z_encoder = z_encoder.reshape(B, H_e, H_e, D_e).permute(0, 3, 1, 2)
            z_encoder = F.interpolate(z_encoder, size=(H_m, H_m), mode='bilinear', align_corners=False)
            z_encoder = z_encoder.permute(0, 2, 3, 1).reshape(B, T_m, D_e)
            z_encoder = F.normalize(z_encoder, dim=-1)
        
        # Negative cosine similarity (we want to maximize similarity)
        loss = -(z_model * z_encoder).sum(dim=-1).mean()
        return loss
    
    # ------------------------------------------------------------------
    # Velocity field helper (shared by all solvers)
    # ------------------------------------------------------------------
    def _velocity(self, z, t_scalar, y, cfg_scale):
        """Evaluate the learned velocity field v_theta(z, t)."""
        if self.use_cond:
            return self.net.forward_with_cfg(z, t_scalar, y, cfg_scale)
        else:
            return self.net(z, t_scalar)

    # ------------------------------------------------------------------
    # Solver 1: Euler (1st order) — 1 NFE per step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_euler(self, batch_size, cfg_scale=5.0, sample_steps=25, return_all_steps=False):
        if self.use_cond:
            y = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        else:
            y = None

        z = torch.randn((batch_size, self.channels, self.image_size, self.image_size), device=self.device)
        dt = 1.0 / sample_steps

        images = [z]
        t_span = torch.linspace(1, 0, sample_steps + 1, device=self.device)
        for i in tqdm(range(sample_steps)):
            t = t_span[i]
            v = self._velocity(z, t, y, cfg_scale)
            z = z - v * dt
            images.append(z)

        z = unnormalize_to_0_1(z.clip(-1, 1))
        if return_all_steps:
            return z, torch.stack(images)
        return z

    # ------------------------------------------------------------------
    # Solver 2: Heun / Improved Euler (2nd order) — 2 NFE per step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_heun(self, batch_size, cfg_scale=5.0, sample_steps=25, return_all_steps=False):
        if self.use_cond:
            y = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        else:
            y = None

        z = torch.randn((batch_size, self.channels, self.image_size, self.image_size), device=self.device)
        dt = 1.0 / sample_steps

        images = [z]
        t_span = torch.linspace(1, 0, sample_steps + 1, device=self.device)
        for i in tqdm(range(sample_steps)):
            t_curr = t_span[i]
            t_next = t_span[i + 1]

            v1 = self._velocity(z, t_curr, y, cfg_scale)
            z_pred = z - v1 * dt

            v2 = self._velocity(z_pred, t_next, y, cfg_scale)
            z = z - 0.5 * (v1 + v2) * dt
            images.append(z)

        z = unnormalize_to_0_1(z.clip(-1, 1))
        if return_all_steps:
            return z, torch.stack(images)
        return z

    # ------------------------------------------------------------------
    # Solver 3: Classical RK4 (4th order) — 4 NFE per step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_rk4(self, batch_size, cfg_scale=5.0, sample_steps=25, return_all_steps=False):
        if self.use_cond:
            y = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        else:
            y = None

        z = torch.randn((batch_size, self.channels, self.image_size, self.image_size), device=self.device)
        dt = 1.0 / sample_steps

        images = [z]
        t_span = torch.linspace(1, 0, sample_steps + 1, device=self.device)
        for i in tqdm(range(sample_steps)):
            t = t_span[i]
            t_mid = t - 0.5 * dt
            t_next = t_span[i + 1]

            k1 = self._velocity(z, t, y, cfg_scale)
            k2 = self._velocity(z - 0.5 * k1 * dt, t_mid, y, cfg_scale)
            k3 = self._velocity(z - 0.5 * k2 * dt, t_mid, y, cfg_scale)
            k4 = self._velocity(z - k3 * dt, t_next, y, cfg_scale)

            z = z - (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
            images.append(z)

        z = unnormalize_to_0_1(z.clip(-1, 1))
        if return_all_steps:
            return z, torch.stack(images)
        return z

    # ------------------------------------------------------------------
    # Solver 4: Adaptive Dormand-Prince (RK45) via torchdiffeq
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_adaptive(self, batch_size, cfg_scale=5.0, atol=1e-5, rtol=1e-5, return_all_steps=False):
        from torchdiffeq import odeint

        if self.use_cond:
            y = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        else:
            y = None

        z0 = torch.randn((batch_size, self.channels, self.image_size, self.image_size), device=self.device)
        shape = z0.shape
        self._adaptive_nfe = 0

        def ode_fn(s, z_flat):
            z = z_flat.reshape(shape)
            t_val = 1.0 - s
            v = self._velocity(z, t_val, y, cfg_scale)
            self._adaptive_nfe += 1
            return v.reshape(-1)

        t_eval = torch.tensor([0.0, 1.0], device=self.device)
        sol = odeint(ode_fn, z0.reshape(-1), t_eval, method='dopri5', atol=atol, rtol=rtol)
        z = sol[-1].reshape(shape)

        z = unnormalize_to_0_1(z.clip(-1, 1))
        return z

    # ------------------------------------------------------------------
    # Unified sample interface
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, batch_size, cfg_scale=5.0, sample_steps=50, solver="euler", return_all_steps=False, **kwargs):
        if solver == "euler":
            return self.sample_euler(batch_size, cfg_scale, sample_steps, return_all_steps)
        elif solver == "heun":
            return self.sample_heun(batch_size, cfg_scale, sample_steps, return_all_steps)
        elif solver == "rk4":
            return self.sample_rk4(batch_size, cfg_scale, sample_steps, return_all_steps)
        elif solver == "adaptive":
            return self.sample_adaptive(batch_size, cfg_scale, **kwargs)
        else:
            raise ValueError(f"Unknown solver: {solver}")

    # ------------------------------------------------------------------
    # Trajectory straightness measurement
    # ------------------------------------------------------------------
    @torch.no_grad()
    def measure_straightness(self, batch_size, cfg_scale=5.0, sample_steps=50):
        """
        Measure how straight the ODE trajectories are.
        Returns the mean cosine similarity between consecutive velocity vectors.
        Straighter trajectories -> cos sim closer to 1 -> Euler is sufficient.
        """
        if self.use_cond:
            y = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        else:
            y = None

        z = torch.randn((batch_size, self.channels, self.image_size, self.image_size), device=self.device)
        dt = 1.0 / sample_steps

        t_span = torch.linspace(1, 0, sample_steps + 1, device=self.device)
        prev_v = None
        cos_sims = []

        for i in tqdm(range(sample_steps), desc="Measuring straightness"):
            t = t_span[i]
            v = self._velocity(z, t, y, cfg_scale)

            if prev_v is not None:
                v_flat = v.reshape(batch_size, -1)
                pv_flat = prev_v.reshape(batch_size, -1)
                cos = F.cosine_similarity(v_flat, pv_flat, dim=-1)
                cos_sims.append(cos.mean().item())

            z = z - v * dt
            prev_v = v

        return cos_sims

    @torch.no_grad()
    def sample_each_class(self, n_per_class, cfg_scale=5.0, sample_steps=50, return_all_steps=False):
        c = torch.arange(self.num_classes, device=self.device).repeat(n_per_class)
        z = torch.randn(self.num_classes * n_per_class, self.channels, self.image_size, self.image_size, device=self.device)
        
        images = []
        t_span = torch.linspace(0, 1, sample_steps, device=self.device)
        for t in tqdm(reversed(t_span)):
            v_t = self.net.forward_with_cfg(z, t, c, cfg_scale)
            z = z - v_t / sample_steps
            images.append(z)
        
        z = unnormalize_to_0_1(z.clip(-1, 1))
        
        if return_all_steps:
            return z, torch.stack(images)
        return z