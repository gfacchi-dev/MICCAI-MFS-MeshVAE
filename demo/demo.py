"""Minimal Marfan-screening demo.

Loads the shared encoder + linear screener (one model, trained on the whole
cohort) and scores a registered facial mesh. With no arguments it scores the two
bundled example faces — the mean of the real Marfan and control meshes (class
averages, not any individual) — so you can see a Marfan-like face score high and
a control-like face score low. Point --mesh at your own mesh registered to the
shared 7160-vertex template to screen a face.

    python demo/demo.py
    python demo/demo.py --mesh my_registered_face.obj --age 22 --sex M
"""
import argparse
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from marfan_face import MarfanScreener  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def score(screener, mesh_path, age, sex):
    verts = np.asarray(trimesh.load(mesh_path, process=False).vertices)
    if verts.shape[0] != 7160:
        raise SystemExit(f"expected 7160 vertices (shared topology), got "
                         f"{verts.shape[0]}; register your mesh to "
                         "weights/template.ply first.")
    out = screener.screen(verts, age_years=age, sex=sex)
    p = out["marfan_probability"]
    tag = "MARFAN-like" if p >= 0.5 else "control-like"
    print(f"  {os.path.basename(mesh_path):24s} age {age:>3.0f}/{sex}  "
          f"|z|={np.linalg.norm(out['latent']):5.2f}  P(Marfan)={p:.3f}  -> {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=None,
                    help="registered .obj/.ply (default: bundled example prototypes)")
    ap.add_argument("--age", type=float, default=25.0, help="age in years")
    ap.add_argument("--sex", default="F", choices=["F", "M"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    screener = MarfanScreener(device=args.device)
    print("screener: single model (logistic regression on 64-d latent)")

    if args.mesh:
        score(screener, args.mesh, args.age, args.sex)
    else:
        score(screener, os.path.join(HERE, "example_marfan.obj"), 22, "M")
        score(screener, os.path.join(HERE, "example_control.obj"), 28, "F")

    print("note: threshold 0.50 is the default operating point; for "
          "high-sensitivity\n      screening use ~0.28 (>=95% recall) as in the paper.")


if __name__ == "__main__":
    main()
