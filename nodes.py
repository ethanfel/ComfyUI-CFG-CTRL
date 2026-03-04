import torch
import torch.nn.functional as F


class SMCCFGCtrl:
    """
    Implements SMC-CFG (Sliding Mode Control CFG) from the paper:
    "CFG-Ctrl: A Control-Theoretic Perspective on Classifier-Free Guidance" (CVPR 2026)
    https://github.com/hanyang-21/CFG-Ctrl

    Replaces standard linear CFG with a nonlinear sliding mode controller
    that prevents instability, overshooting, and artifacts at high guidance scales.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "smc_cfg_lambda": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 50.0, "step": 0.01,
                    "tooltip": "Sliding surface coefficient. Controls how much the controller weights previous error magnitude vs error derivative. Paper recommended: 5.0",
                }),
                "smc_cfg_K": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Switching gain. Bounds the correction to [-K, +K] per element. Higher = stronger correction but may introduce chattering. Paper recommended: 0.2",
                }),
                "warmup_steps": ("INT", {
                    "default": 0, "min": 0, "max": 100,
                    "tooltip": "Number of initial steps with no guidance (pure conditional prediction). Lets the model establish structure before guidance kicks in.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "sampling/custom_sampling"

    def patch(self, model, smc_cfg_lambda, smc_cfg_K, warmup_steps):
        state = {"prev_eps": None, "step": 0, "prev_sigma": None}
        lam = smc_cfg_lambda
        K = smc_cfg_K

        def smc_cfg_function(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            sigma = args["sigma"]

            # Detect new generation (sigma jumps up = new sampling run)
            curr_sigma = sigma.max().item() if torch.is_tensor(sigma) else float(sigma)
            if state["prev_sigma"] is not None and curr_sigma > state["prev_sigma"] * 1.1:
                state["prev_eps"] = None
                state["step"] = 0
            state["prev_sigma"] = curr_sigma

            step = state["step"]
            state["step"] = step + 1

            if warmup_steps > 0 and step < warmup_steps:
                return cond

            # Normalize to noise-prediction space (divide out sigma).
            sigma_val = max(curr_sigma, 1e-8)
            guidance_eps = (cond - uncond) / sigma_val

            if state["prev_eps"] is None:
                state["prev_eps"] = guidance_eps.detach().clone()

            prev_eps = state["prev_eps"]

            # Sliding surface: s_t = (e_t - e_{t-1}) + lambda * e_{t-1}
            s = (guidance_eps - prev_eps) + lam * prev_eps

            # Scale K so the output correction (cond_scale * K_eff) stays
            # in the range the paper tested (0.5–1.5). The paper only
            # tested up to cfg=7.5; at cfg=12, full K gives cond_scale*K=2.4
            # which corrupts the image. sqrt scaling keeps the output
            # correction growing sub-linearly with cfg.
            K_eff = K / max(cond_scale, 1.0) ** 0.5

            # Smooth switching via tanh instead of sign to avoid
            # salt-and-pepper artifacts in ComfyUI's latent space.
            phi = s.std().clamp(min=1e-6)
            u_sw = -K_eff * torch.tanh(s / phi)

            # Spatial blur to smooth grid artifacts at VAE patch boundaries.
            if u_sw.ndim == 4:
                u_sw = F.avg_pool2d(u_sw, kernel_size=5, stride=1, padding=2)

            # Store RAW guidance (before correction) to prevent correction
            # accumulation through the lambda * prev_eps term.
            state["prev_eps"] = guidance_eps.detach().clone()

            return uncond + cond_scale * (guidance_eps + u_sw) * sigma_val

        m = model.clone()
        m.set_model_sampler_cfg_function(smc_cfg_function, disable_cfg1_optimization=True)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "SMCCFGCtrl": SMCCFGCtrl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMCCFGCtrl": "SMC-CFG Ctrl",
}
