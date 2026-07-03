"""Review-sanctioned mixed precision: bf16 trunk + fp32 readout head, computed OUTSIDE the model.

The model runs pure bf16 (no in-graph dtype mixing — MPS asserts on that). We capture the final
residual and apply ln_final + the two answer-token unembed columns + Gemma's logit softcap in fp32.
Replaces FeatureScope._readout via attach(); everything downstream (_ensure_base, _dir_shift,
nulls) inherits it.
"""
import copy, types, torch

def attach(fs):
    model = fs.model
    ln32 = copy.deepcopy(model.ln_final).to(torch.float32)
    W2 = model.unembed.W_U[:, [int(fs._pos), int(fs._neg)]].detach().float().clone()
    b2 = model.unembed.b_U[[int(fs._pos), int(fs._neg)]].detach().float().clone() if model.unembed.b_U is not None else None
    cap = getattr(model.cfg, "output_logits_soft_cap", None)
    final_hook = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"

    def _readout(self, text, steer=None):
        hooks = [(self.hook, lambda r, hook: r + steer)] if steer is not None else []
        grab = {}
        hooks = hooks + [(final_hook, lambda r, hook: grab.__setitem__("r", r[0, -1].detach()))]
        with torch.no_grad():
            self.model.run_with_hooks(
                self.model.to_tokens(self._few + self.concept.template.format(text=text)),
                fwd_hooks=hooks, return_type=None)
            x = ln32(grab["r"].float().unsqueeze(0)).squeeze(0).float()
            lg = x @ W2 + (b2 if b2 is not None else 0.0)
            if cap:
                lg = cap * torch.tanh(lg / cap)
        return float(lg[0] - lg[1])

    fs._readout = types.MethodType(_readout, fs)
    return fs
