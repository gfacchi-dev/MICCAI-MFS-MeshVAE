"""End-to-end Marfan screening from a registered facial mesh.

Loads the shared encoder + linear screener (one model, trained on the whole
cohort) and exposes a single call:
    MarfanScreener().screen(vertices, age_years, sex) -> P(Marfan)

The pipeline is: normalize vertices -> encode to the 64-d posterior mean ->
apply the logistic-regression screener. No face is ever reconstructed.
"""
import json
import os
import pickle

import numpy as np
import torch

from .encoder import SpiralEncoder

_HERE = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS = os.path.normpath(os.path.join(_HERE, "..", "weights"))


class MarfanScreener:
    def __init__(self, weights_dir=_WEIGHTS, device="cpu"):
        self.device = torch.device(device)
        meta = json.load(open(os.path.join(weights_dir, "metadata.json")))
        self.meta = meta

        topo = os.path.join(weights_dir, "topology")
        with open(os.path.join(topo, "spirals.pkl"), "rb") as f:
            spiral_indices = pickle.load(f)
        with open(os.path.join(topo, "down_transforms.pkl"), "rb") as f:
            down_transforms = pickle.load(f)
        norm = torch.load(os.path.join(weights_dir, "norm.pt"), weights_only=False)
        self.vmean = norm["mean"].to(self.device).float()
        self.vstd = norm["std"].to(self.device).float()

        self.encoder = SpiralEncoder(
            in_channels=meta["in_channels"],
            out_channels=meta["out_channels"],
            latent_size=meta["latent_size"],
            spiral_indices=spiral_indices,
            down_transforms=down_transforms,
            cond_dim=meta["cond_dim"],
        )
        state = torch.load(os.path.join(weights_dir, "encoder.pt"), weights_only=False)
        self.encoder.load_state_dict(state, strict=True)
        self.encoder.to(self.device).eval()

        ast = torch.load(os.path.join(weights_dir, "age_stats.pt"), weights_only=False)
        self.age_mean, self.age_std = float(ast["mean"]), float(ast["std"])

        with open(os.path.join(weights_dir, "screener.pkl"), "rb") as f:
            self.screener = pickle.load(f)   # sklearn LogisticRegression

    # --- conditioning: [standardized age (months), sex] --------------------
    def _cond(self, age_years, sex):
        parts = []
        if self.meta["condition_age"]:
            am = float(age_years) * 12.0
            if np.isnan(am):
                am = self.age_mean
            parts.append((am - self.age_mean) / self.age_std)
        if self.meta["condition_sex"]:
            s = str(sex).strip().upper()
            parts.append(1.0 if s == "M" else 0.0 if s == "F" else 0.5)
        return torch.tensor([parts], dtype=torch.float, device=self.device)

    @torch.no_grad()
    def embed(self, vertices, age_years, sex):
        """vertices: (V,3) array in the shared template topology. Returns (d_z,)."""
        v = torch.as_tensor(np.asarray(vertices), dtype=torch.float,
                            device=self.device)
        v = ((v - self.vmean) / self.vstd).unsqueeze(0)
        z = self.encoder.encode(v, self._cond(age_years, sex))
        return z[0].cpu().numpy()

    def screen(self, vertices, age_years, sex):
        """Returns dict with Marfan probability and the deterministic latent."""
        z = self.embed(vertices, age_years, sex)
        p = float(self.screener.predict_proba(z[None])[0, 1])
        return {"marfan_probability": p, "latent": z}
