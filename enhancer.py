import os
import torch
from gfpgan import GFPGANer
from PIL import Image
import numpy as np

def enhance_face(input_path, output_path, model_path='weights/GFPGANv1.4.pth'):
    restorer = GFPGANer(
        model_path=model_path,
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None
    )

    img = Image.open(input_path).convert('RGB')
    _, _, restored_img = restorer.enhance(np.array(img), has_aligned=False, only_center_face=False, paste_back=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(restored_img).save(output_path)
    print(f"[✔] Saved enhanced face to: {output_path}")
