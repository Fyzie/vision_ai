import torch
import cv2
import numpy as np
import os
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

IMAGE_PATH = "C:/Users/Hafizi/Documents/computer vision/03-lens/display_1.png" 

device = "cuda" if torch.cuda.is_available() else "cpu"

model_cfg = "C:/Users/Hafizi/miniconda3/envs/rfdetr-1.6/Lib/site-packages/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
sam2_checkpoint = "C:/Users/Hafizi/Documents/computer vision/sam2.1_hiera_tiny.pt"

try:
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
except Exception as e:
    print(f"Error loading model")
    exit()

def process_image(image_path):
    frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        print(f"Cannot load {image_path}")
        return

    if frame.dtype != np.uint8:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if len(frame.shape) == 2:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 1:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4: # Handle PNG with Alpha
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        frame_rgb = frame

    try:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=dtype):
            masks = mask_generator.generate(frame_rgb)
    except Exception as e:
        print(f"Inference error: {e}")
        return frame_rgb

    overlay = frame_rgb.copy()
    if masks:
        for mask in masks:
            m = mask['segmentation']
            color = np.random.randint(0, 255, (3,)).tolist()
            overlay[m] = color

    result = cv2.addWeighted(frame_rgb, 0.6, overlay, 0.4, 0)
    return result

if os.path.exists(IMAGE_PATH):
    output = process_image(IMAGE_PATH)
    
    if output is not None:
        display_scale = 0.8
        h, w = output.shape[:2]
        display_img = cv2.resize(output, (int(w * display_scale), int(h * display_scale)))
        
        cv2.imshow('SAM Inference', display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print(f"No file")
