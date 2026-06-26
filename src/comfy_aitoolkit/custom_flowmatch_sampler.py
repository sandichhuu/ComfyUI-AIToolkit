try:
    import math
    from typing import Union
    from torch.distributions import LogNormal
    from diffusers import FlowMatchEulerDiscreteScheduler
    import torch
    import numpy as np
except (ImportError, ModuleNotFoundError):
    pass

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

class CustomFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        # Tạo một config giả lập để tránh lỗi nếu Diffusers đòi hỏi config object
        super().__init__(*args, **kwargs)
        self.init_noise_sigma = 1.0
        self.timestep_type = "linear"

    def set_custom_inference_sigmas(self, num_inference_steps, timestep_type, latents=None, patch_size=1):
        device = "cpu"
        self.timestep_type = timestep_type
        
        if timestep_type in ['linear', 'weighted']:
            sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)
            return torch.from_numpy(sigmas).to(dtype=torch.float32)

        elif timestep_type == 'sigmoid':
            t = torch.sigmoid(torch.randn((num_inference_steps,), device=device))
            timesteps = ((1 - t) * 1000)
            timesteps, _ = torch.sort(timesteps, descending=True)
            sigmas = timesteps / 1000.0
            sigmas = torch.cat([sigmas, torch.zeros(1)])
            return sigmas
            
        elif timestep_type in ['flux_shift', 'lumina2_shift', 'shift']:
            timesteps = np.linspace(1000.0, 1.0, num_inference_steps)
            sigmas = timesteps / 1000.0

            # Đọc config nội bộ được truyền từ constructor thông qua kwargs (đã lưu vào self.config)
            use_dynamic_shifting = self.config.get("use_dynamic_shifting", True)

            if use_dynamic_shifting:
                if latents is None or "samples" not in latents:
                    image_seq_len = 1024 
                else:
                    h = latents["samples"].shape[2]
                    w = latents["samples"].shape[3]
                    image_seq_len = h * w // (patch_size**2)

                mu = calculate_shift(
                    image_seq_len,
                    self.config.get("base_image_seq_len", 256),
                    self.config.get("max_image_seq_len", 4096),
                    self.config.get("base_shift", 0.5),
                    self.config.get("max_shift", 1.16),
                )
                sigmas = mu * sigmas / (1 + (mu - 1) * sigmas)
            else:
                shift = self.config.get("shift", 1.0)
                sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

            sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
            
            if sigmas[-1] != 0.0:
                sigmas = torch.cat([sigmas, torch.zeros(1)])
            return sigmas

        elif timestep_type == 'lognorm_blend':
            alpha = 0.75
            lognormal = LogNormal(loc=0, scale=0.333)
            t1 = lognormal.sample((int(num_inference_steps * alpha),)).to(device)
            t1 = ((1 - t1/t1.max()) * 1000)
            t2 = torch.linspace(1000, 1, int(num_inference_steps * (1 - alpha)), device=device)
            timesteps = torch.cat((t1, t2))
            timesteps, _ = torch.sort(timesteps, descending=True)
            
            sigmas = timesteps / 1000.0
            sigmas = torch.cat([sigmas, torch.zeros(1)])
            return sigmas
            
        else:
            sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)
            return torch.from_numpy(sigmas).to(dtype=torch.float32)