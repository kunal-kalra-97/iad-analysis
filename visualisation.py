"""
PCA-to-RGB feature maps for layer1..layer4 across a set of images, so we
can SEE the resolution v semantics tradeoff and how different defect types look
at different depths.

"""

import sys
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from feature_extractor import WideResNet50FeatureExtractor, build_transform, LAYERS

# IMAGE_PATHS = [
#     "data/transistor/test/good/000.png",
#     "data/transistor/test/bent_lead/000.png",
#     "data/transistor/test/misplaced/000.png",
#     "data/transistor/test/cut_lead/000.png",
# ]

IMAGE_PATHS = [
    "data/bottle/test/good/000.png",
    "data/bottle/test/broken_large/000.png",
    "data/bottle/test/broken_small/000.png",
    "data/bottle/test/contamination/000.png",
]


@torch.no_grad()
def extract_all(model, transform, paths):
    extracted = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        x = transform(img).unsqueeze(0)                 # [1,3,224,224]
        feats = model(x)                                # {layer: [1,C,h,w]}
        feats = {k: v[0].cpu().numpy() for k, v in feats.items()}
        extracted.append((img, feats))
    return extracted


def pca_rgb_per_layer(extracted, layer, n_components=3):
    mats, shapes = [], []
    for _, feats in extracted:
        fm = feats[layer]                               # [C,h,w]
        C, h, w = fm.shape
        mats.append(fm.reshape(C, h * w).T)             # [h*w, C]
        shapes.append((h, w))
    pooled = np.concatenate(mats, axis=0)               # [sum hw, C]

    mu = pooled.mean(0, keepdims=True)
    sd = pooled.std(0, keepdims=True) + 1e-6
    pooled = (pooled - mu) / sd

    proj = PCA(n_components=n_components, random_state=0).fit_transform(pooled)

    pmin = proj.min(0, keepdims=True)
    pmax = proj.max(0, keepdims=True)
    proj = (proj - pmin) / (pmax - pmin + 1e-6)         # [sum hw, 3] in [0,1]

    rgbs, idx = [], 0
    for (h, w) in shapes:
        n = h * w
        rgbs.append(proj[idx: idx + n].reshape(h, w, 3))
        idx += n
    return rgbs


def upsample_nearest(rgb, size=224):
    im = Image.fromarray((rgb * 255).astype(np.uint8))
    return np.asarray(im.resize((size, size), Image.NEAREST)) / 255.0


def make_figure(extracted, layers=LAYERS, out_path="feature_pca_grid.png"):
    per_layer = {L: pca_rgb_per_layer(extracted, L) for L in layers}
    n, ncol = len(extracted), 1 + len(layers)
    fig, axes = plt.subplots(n, ncol, figsize=(2.4 * ncol, 2.4 * n))
    axes = np.atleast_2d(axes)
    titles = ["input"] + [f"{L} (PCA-RGB)" for L in layers]
    for j, t in enumerate(titles):
        axes[0, j].set_title(t, fontsize=10)
    for i, (img, _) in enumerate(extracted):
        axes[i, 0].imshow(img.resize((224, 224)))
        axes[i, 0].axis("off")
        for j, L in enumerate(layers):
            axes[i, j + 1].imshow(upsample_nearest(per_layer[L][i]))
            axes[i, j + 1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved {out_path}  ({n} images x {len(layers)} layers)")
    return out_path


if __name__ == "__main__":
    paths = sys.argv[1:] or IMAGE_PATHS
    model = WideResNet50FeatureExtractor()
    transform = build_transform()
    extracted_feats = extract_all(model, transform, paths)
    make_figure(extracted_feats, layers=LAYERS, out_path="output/bottle/feature_pca_grid.png")