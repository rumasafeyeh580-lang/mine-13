from abc import ABC, abstractmethod
from PIL import Image
import time
from typing import Iterable
from config.settings import BackgroundRemovalConfig

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image, resized_crop
from logger_config import logger

class BaseBGRemover(ABC):
    def __init__(self, settings: BackgroundRemovalConfig) -> None:
        self._image_threshold_check = 500
        self.settings = settings

        self.device = f"cuda:{settings.gpu}" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def unload_model(self) -> None:
        pass

    @abstractmethod
    def remove_bg(self, image: Image) -> tuple[Image, bool]:
        pass
    
    async def startup(self) -> None:
        """
        Startup the BaseBGRemover.
        """
        # Load model
        try:
            self.load_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Error loading model: {e}")

    async def shutdown(self) -> None:
        """
        Shutdown the BaseBGRemover.
        """
        logger.info("BaseBGRemover closed.")

    def is_image_valid(self, image: Image) -> bool:
        """ Function that checks if the image after background removal is empty or barely filled in with image data. """

        # Convert the image to grayscale
        img_gray = image.convert('L')

        # Convert the grayscale image to a NumPy array
        img_array = np.array(img_gray)

        # Calculate the variance of pixel values
        variance = np.var(img_array)

        if variance > self._image_threshold_check:
            return True
        else:
            return False

    def filter_semi_transparent_pixels(self, image: Image.Image) -> Image.Image:
        # Convert to numpy array
        img_np = np.array(image)

        # Extract alpha channel (4th channel)
        alpha = img_np[:, :, 3] / 255.0
        mask = alpha >= 0.72

        # Apply mask to all 4 channels
        filtered_np = img_np.copy()
        filtered_np[~mask] = [0, 0, 0, 0]  # Set fully transparent or black

        # Convert back to PIL image
        filtered_img = Image.fromarray(filtered_np, mode="RGBA")
        return filtered_img

    def _crop_and_center(self, tensor_rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Remove the background from the image.
        """

        # Normalize tensor value for background removal model, reshape for model batch processing (C=3, H, W) -> (1, C=3, H, W)

        # Get bounding box indices
        bbox_indices = torch.argwhere(mask > 0.8)
        if len(bbox_indices) == 0:
            crop_args = dict(top = 0, left = 0, height = mask.shape[1], width = mask.shape[0])
        else:
            h_min, h_max = torch.aminmax(bbox_indices[:, 1])
            w_min, w_max = torch.aminmax(bbox_indices[:, 0])
            width, height = w_max - w_min, h_max - h_min
            center =  (h_max + h_min) / 2, (w_max + w_min) / 2
            size = max(width, height)
            padded_size_factor = 1 + self.padding_percentage
            size = int(size * padded_size_factor)

            top = int(center[1] - size // 2)
            left = int(center[0] - size // 2)
            bottom = int(center[1] + size // 2)
            right = int(center[0] + size // 2)

            if self.limit_padding:
                top = max(0, top)
                left = max(0, left)
                bottom = min(mask.shape[1], bottom)
                right = min(mask.shape[0], right)
            
            crop_args = dict(
                top=top,
                left=left,
                height=bottom - top,
                width=right - left
            )
        

        mask = mask.unsqueeze(0)
        # Concat mask with image and blacken the background: (C=3, H, W) | (1, H, W) -> (C=4, H, W)
        tensor_rgba = torch.cat([tensor_rgb*mask, mask], dim=-3)
        output = resized_crop(tensor_rgba, **crop_args, size = self.output_size, antialias=False)
        return output
    
    def remove_background(self, image: Image.Image | Iterable[Image.Image]) -> Image.Image | Iterable[Image.Image]:
        """
        Remove the background from the image.
        """
        # try:
        t1 = time.time()

        images = image if isinstance(image, Iterable) else [image]

        outputs = []
        has_alpha = False

        for img in images:
            if img.mode == "RGBA":
                # Get alpha channel
                alpha = np.array(img)[:, :, 3]
                if not np.all(alpha==255):
                    has_alpha=True
            
            if has_alpha:
                # If the image has alpha channel, return the image
                output = img
                
            else:
                # PIL.Image (H, W, C) C=3
                # Tensor (H, W, C) -> (C, H',W')
                # rgb_tensor = self.transforms(rgb_image).to(self.device)
                tensor_rgb, mask = self.remove_bg(img)
                output = self._crop_and_center(tensor_rgb, mask)

            outputs.append(output)

        images_without_background = tuple(to_pil_image(o[:3]) for o in outputs) if isinstance(image, Iterable) else to_pil_image(outputs[0][:3])

        removal_time = time.time() - t1
        logger.success(f"Background remove - Time: {removal_time:.2f}s - Images without background: {len(images_without_background)}")

        return images_without_background