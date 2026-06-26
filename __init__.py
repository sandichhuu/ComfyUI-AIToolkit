"""Top-level package for comfy_aitoolkit."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS"
]

__author__ = """sandichhuu"""
__email__ = "sandichhuu@gmail.com"
__version__ = "0.0.1"

from .src.comfy_aitoolkit.nodes import NODE_CLASS_MAPPINGS
from .src.comfy_aitoolkit.nodes import NODE_DISPLAY_NAME_MAPPINGS

# flow_match_name = "[AI-Toolkit] Z-Image-Turbo FlowMatch"

# zit_config = {
#     "num_train_timesteps": 1000,
#     "use_dynamic_shifting": False,
#     "shift": 3.0,
# }

# def zit_scheduler_handler(model_sampling, steps):
#     scheduler = CustomFlowMatchEulerDiscreteScheduler(**zit_config)
#     scheduler.set_timesteps(steps, device=model_sampling.device if hasattr(model_sampling, 'device') else 'cpu', mu=0.0)
#     sigmas = scheduler.sigmas
#     return sigmas

# try:
#     from comfy.samplers import SchedulerHandler, SCHEDULER_HANDLERS, SCHEDULER_NAMES
#     if flow_match_name not in SCHEDULER_HANDLERS:
#         handler = SchedulerHandler(handler=zit_scheduler_handler, use_ms=True)
#         SCHEDULER_HANDLERS[flow_match_name] = handler
#         SCHEDULER_NAMES.append(flow_match_name)
# except ImportError:
#     pass

# # try:
# #     from comfy.samplers import KSampler
# #     if flow_match_name not in KSampler.SCHEDULERS:
# #         KSampler.SCHEDULERS.append(flow_match_name)
# # except ImportError:
# #     pass