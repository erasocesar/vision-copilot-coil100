"""Referencia moderna: embeddings de ResNet-50 preentrenada en ImageNet (sin entrenar nada)."""
import numpy as np


def embed(images: np.ndarray, batch_size: int = 64) -> np.ndarray:
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    preprocess = weights.transforms()
    out = []
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            x = torch.from_numpy(images[i:i + batch_size]).permute(0, 3, 1, 2)
            out.append(model(preprocess(x)).numpy())
    return np.concatenate(out).astype(np.float32)
