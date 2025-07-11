import os
import cv2
import torch
import numpy as np
from gfpgan import GFPGANer
from PIL import Image

def run_inference(
    input_image_path,
    output_image_path,
    model_path='weights/GFPGANv1.4.pth',
    upscale=2
):
    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None
    )

    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    _, _, output = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, output)
    print(f'[✔] Saved enhanced image to: {output_image_path}')

if __name__ == '__main__':
    input_path = 'inputs/a.png'
    output_path = 'results/'
    run_inference(input_path, output_path)
