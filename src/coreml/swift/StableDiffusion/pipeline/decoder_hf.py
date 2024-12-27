import os
import torch
import numpy as np
import uuid

from PIL import Image
from diffusers import AutoencoderTiny
from diffusers.models import AutoencoderKL

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

class Decoder():
    def __init__(self):
        vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16).to("mps")
        # vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("mps")
        self.vae = vae

    def decode_images(self, latents):
        os.makedirs("tmp", exist_ok=True)
        for latent in latents:
            latent = torch.from_numpy(latent).to(dtype=torch.float16, device="mps")
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latent.device, latent.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latent.device, latent.dtype)
                )
                latent = latent * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latent = latent / self.vae.config.scaling_factor
            image = self.vae.decode(latent, return_dict=False)[0]
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            image = numpy_to_pil(image)
            random_name = f"tmp/{str(uuid.uuid4().hex)}.png"
            image[0].save(random_name)


decoder = Decoder()
