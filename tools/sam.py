import torch
import numpy as np
from PIL import Image
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
from tools import BaseTool
import os
import time

class SAMTool(BaseTool):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.predictor = None
        self.model_type = config.get('model_type', 'vit_h')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.multimask_output = config.get('multimask_output', True)
        self.checkpoint = config.get('checkpoint', None)
        if self.checkpoint is not None:
            self.load_model(self.checkpoint)

    def load_model(self, checkpoint_path: str):
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint_path}")
        self.model = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)

    def process(
        self,
        image: Image.Image,
        subtask_name: str,
        bounding_boxes: list,
        random_color: bool = True,
        alpha: float = 0.6
    ) -> dict:
     
        if self.predictor is None:
            if self.checkpoint is None:
                raise RuntimeError("No checkpoint path provided. Please specify one in config.")
            self.load_model(self.checkpoint)

        image_np = np.array(image)
        self.predictor.set_image(image_np)

        boxes_torch = torch.tensor(bounding_boxes, dtype=torch.float, device=self.device)
        if boxes_torch.ndim != 2 or boxes_torch.shape[1] != 4:
            raise ValueError("bounding_boxes must be a list of [x1, y1, x2, y2] entries.")

        boxes_torch = self.predictor.transform.apply_boxes_torch(boxes_torch, image_np.shape[:2])

        start_time = time.time()
        masks, scores, logits = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=boxes_torch,
            multimask_output=self.multimask_output
        )
        end_time = time.time()

        if self.multimask_output:
            best_masks = masks[:, 0, :, :] 
        else:
            best_masks = masks.squeeze(1)

        final_image, cutout_images = overlay_and_generate_cutouts(image, best_masks.cpu().numpy(), subtask_name, random_color, alpha)

        # output_dir = "outputs"
        # os.makedirs(output_dir, exist_ok=True)

        # cutout_paths = []
        # for idx, cutout_img in enumerate(cutout_images):
        #     cutout_path = os.path.join(output_dir, f"cutout_image_{idx}.png")
        #     cutout_img.save(cutout_path)
        #     cutout_paths.append(cutout_path)
        #     print(f"Cutout {idx} saved at: {cutout_path}")

        execution_time = end_time - start_time 

        return {"image": final_image, "cutout_images": cutout_images, "execution_time": execution_time}




def overlay_and_generate_cutouts(
    original_image: Image.Image,
    mask_tensor: np.ndarray,
    subtask_name: str,
    random_color: bool = False,
    alpha: float = 0.4
) -> tuple:

    image_rgba = original_image.convert("RGBA")
    image_np = np.array(image_rgba)

    B, H, W = mask_tensor.shape  
    if image_np.shape[0] != H or image_np.shape[1] != W:
        raise ValueError("Mask dimension does not match image dimension.")

    modified_image = image_np.copy()
    cutout_images = []  
    for b_idx in range(B):
        single_mask = mask_tensor[b_idx, :, :]

        if single_mask.dtype == np.bool_:
            single_mask = single_mask.astype(np.uint8) * 255

        mask_bool = single_mask > 0  
        cutout_np = np.zeros_like(image_np, dtype=np.uint8)
        cutout_np[mask_bool] = image_np[mask_bool]  
        cutout_np[~mask_bool, 3] = 0 

        cutout_image = Image.fromarray(cutout_np)
        cutout_images.append(cutout_image)

        if subtask_name.lower() == "object segmentation":
            if random_color:
                color = np.random.randint(0, 256, size=3).tolist()  # random RGB
            else:
                color = [30, 144, 255]  # DodgerBlue

            overlay_color = np.array(color + [int(alpha * 255)])

            ys, xs = np.where(mask_bool)
            modified_image[ys, xs, :3] = (
                (1 - alpha) * modified_image[ys, xs, :3].astype(np.float32)
                + alpha * overlay_color[:3]
            ).astype(np.uint8)
            modified_image[ys, xs, 3] = 255  # Fully opaque

        else:
            modified_image[mask_bool, :3] = [255, 255, 255]  # White background
            modified_image[mask_bool, 3] = 0  # Fully transparent

    return Image.fromarray(modified_image), cutout_images
