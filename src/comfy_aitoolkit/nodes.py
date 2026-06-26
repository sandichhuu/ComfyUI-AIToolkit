import comfy.model_management
import comfy.sample
import torch

from .flow_match_preset import FlowMatchPresetConfig
from .custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

class AI_Tookit_Sigmas:
    """
    ai-toolkit sigmas generator.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "model_preset": (list(FlowMatchPresetConfig.MODEL_PRESETS.keys()),),
                "timestep_type": (["linear", "weighted", "sigmoid", "flux_shift", "lumina2_shift", "shift", "lognorm_blend"], {"default": "linear"}),
            },
            "optional": {
                "latents": ("LATENT",),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "execute"
    CATEGORY = "🐲 SAN/🐲 AI-Toolkit"

    def execute(self, steps, model_preset, timestep_type, latents=None):
        selected_config = FlowMatchPresetConfig.MODEL_PRESETS[model_preset].copy()
        scheduler = CustomFlowMatchEulerDiscreteScheduler.from_config(selected_config)
        sigmas = scheduler.set_custom_inference_sigmas(
            num_inference_steps=steps,
            timestep_type=timestep_type,
            latents=latents
        )
        
        return (sigmas,)

class Z_Image_Sampler:
    """
    ai-toolkit z-image sampler.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "timestep_type": (["linear", "weighted", "sigmoid", "flux_shift", "lumina2_shift", "shift", "lognorm_blend"], {"default": "linear"}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"
    CATEGORY = "🐲 SAN/🐲 AI-Toolkit"

    def execute(self, model, positive, negative, latent_image, timestep_type, steps, cfg, seed, denoise):
            selected_config = FlowMatchPresetConfig.MODEL_PRESETS["Z-Image | Turbo"].copy()
            scheduler = CustomFlowMatchEulerDiscreteScheduler.from_config(selected_config)
            sigmas = scheduler.set_custom_inference_sigmas(
                num_inference_steps=steps,
                timestep_type=timestep_type,
                latents=latent_image,
            )

            device = comfy.model_management.get_torch_device()
            sigmas = sigmas.to(device)

            latent_samples = latent_image["samples"]
            if denoise < 1.0:
                sigmas = sigmas[int(len(sigmas) * (1.0 - denoise)):]
            
            sampled = comfy.sample.sample(
                model, 
                noise=torch.zeros_like(latent_samples), 
                steps=steps, 
                cfg=cfg, 
                sampler_name="euler_ancestral",
                scheduler="custom",
                positive=positive, 
                negative=negative, 
                latent_image=latent_samples,
                denoise=denoise,
                sigmas=sigmas, 
                seed=seed
            )

            out = latent_image.copy()
            out["samples"] = sampled
            return (out,)

NODE_CLASS_MAPPINGS = {
    "🐲 [AI-Toolkit] Sigmas":           AI_Tookit_Sigmas,
    "🐲 [AI-Toolkit] Z-Image Sampler":  Z_Image_Sampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "🐲 [AI-Toolkit] Sigmas":           "🐲 [AI-Toolkit] Sigmas",
    "🐲 [AI-Toolkit] Z-Image Sampler":  "🐲 [AI-Toolkit] Z-Image Sampler"
}