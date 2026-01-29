# gradcam_utils.py
import cv2
import numpy as np
import torch
from torch.nn import functional as F

def generate_gradcam(model, image_tensor, target_class):
    model.eval()
    gradients = []
    activations = []

    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activations(module, input, output):
        activations.append(output)

    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module

    last_conv.register_forward_hook(save_activations)
    last_conv.register_backward_hook(save_gradients)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    class_loss = output[0, target_class]
    class_loss.backward()

    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    return cam
