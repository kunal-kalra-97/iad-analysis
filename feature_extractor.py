from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

LAYERS = ("layer1", "layer2", "layer3", "layer4")

EXPECTED = {
    "layer1": (256, 56),
    "layer2": (512, 28),
    "layer3": (1024, 14),
    "layer4": (2048, 7),
}

class WideResNet50FeatureExtractor(nn.Module):
    def __init__(self, layers = LAYERS, backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)):
        super().__init__()
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone
        self.layers = tuple(layers)
        self._features = OrderedDict()
        self._handles = []
        for name in self.layers:
            module = getattr(self.backbone, name)
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(_module, _inp, out):
            self._features[name] = out
        return hook

    @torch.no_grad()
    def forward(self, x):
        # x: [B, 3, H, W]
        self._features = OrderedDict()
        self.backbone(x)
        return OrderedDict(
            (name, self._features[name].detach().clone()) for name in self.layers
        )

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

def build_transform(resize=256, crop=224):
    # resize, center crop, ImageNet norm.
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


if __name__ == "__main__":
    model = WideResNet50FeatureExtractor()
    dummy = torch.randn(2, 3, 224, 224)
    feats = model(dummy)

    all_ok = True
    for name, t in feats.items():
        c, hw = EXPECTED[name]
        ok = (t.shape[1] == c and t.shape[2] == hw and t.shape[3] == hw)
        all_ok &= ok
        # layer name, shape, expected(c, hw)
        print(f"{name} {t.shape}    {(c, hw)}     {'PASS' if ok else 'FAIL'}")

    assert all_ok, "shape check failed -- hooks are grabbing the wrong tensors"
    assert not model.backbone.training, "backbone must be in eval() mode"
    assert all(not p.requires_grad for p in model.backbone.parameters()), "backbone params must be frozen"
    print("\nall checks passed: hooks fire, shapes correct, backbone frozen + eval")
