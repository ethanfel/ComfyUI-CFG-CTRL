import torch


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
        # Mutable state persisted across denoising steps via closure
        state = {
            "prev_eps": None,
            "step": 0,
            "prev_sigma": None,
        }

        lam = smc_cfg_lambda
        K = smc_cfg_K

        def smc_cfg_function(args):
            # Use denoised-space predictions — these have consistent magnitude
            # across sigma values. ComfyUI's args["cond"]/["uncond"] are
            # (x - denoised), which are sigma-scaled and would make the fixed
            # K correction dominate at low sigma (late steps), destroying the image.
            cond_denoised = args["cond_denoised"]
            uncond_denoised = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            x = args["input"]
            sigma = args["sigma"]

            # Detect new generation: sigma should decrease monotonically during
            # denoising. If it jumps up, a new sampling run has started.
            curr_sigma = sigma.max().item() if torch.is_tensor(sigma) else float(sigma)
            if state["prev_sigma"] is not None and curr_sigma > state["prev_sigma"] * 1.1:
                state["prev_eps"] = None
                state["step"] = 0
            state["prev_sigma"] = curr_sigma

            step = state["step"]
            state["step"] = step + 1

            # Warmup: pure conditional prediction (no guidance)
            if warmup_steps > 0 and step < warmup_steps:
                return x - cond_denoised

            # Guidance error in denoised space (consistent magnitude across sigma)
            guidance_eps = cond_denoised - uncond_denoised

            # Initialize prev_eps on first SMC step (matches original paper
            # where SMC correction is applied from the very first step)
            if state["prev_eps"] is None:
                state["prev_eps"] = guidance_eps.detach().clone()

            prev_eps = state["prev_eps"]

            # Sliding surface: s_t = (e_t - e_{t-1}) + lambda * e_{t-1}
            s = (guidance_eps - prev_eps) + lam * prev_eps

            # Switching control: u_sw = -K * sign(s_t)
            u_sw = -K * torch.sign(s)

            # Corrected guidance error
            guidance_eps = guidance_eps + u_sw

            # Store corrected guidance for next step's sliding surface
            state["prev_eps"] = guidance_eps.detach().clone()

            # Guided denoised output
            denoised = uncond_denoised + cond_scale * guidance_eps

            # Return noise residual (framework computes cfg_result = x - return)
            return x - denoised

        m = model.clone()
        m.set_model_sampler_cfg_function(smc_cfg_function, disable_cfg1_optimization=True)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "SMCCFGCtrl": SMCCFGCtrl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMCCFGCtrl": "SMC-CFG Ctrl",
}
