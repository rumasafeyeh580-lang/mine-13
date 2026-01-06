import gc
from PIL import Image

import torch
from modules.bg_removers.ben2_model.ben2 import BEN_Base
from modules.bg_removers.base_bg_remover import BaseBGRemover
from config.settings import BackgroundRemovalConfig
from torchvision import transforms

class Ben2BGRemover(BaseBGRemover):
    def __init__(self, settings: BackgroundRemovalConfig):
        super().__init__(settings)
        self._bg_remover: BEN_Base | None = None

        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.settings.input_image_size), 
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )

    def load_model(self) -> None:
        self._bg_remover = BEN_Base.from_pretrained("PramaLLC/BEN2").to(self.device)
        self._bg_remover.eval().half()

    def unload_model(self) -> None:
        del self._bg_remover
        gc.collect()
        torch.cuda.empty_cache()
        self._bg_remover = None

    def remove_bg(self, image: Image) -> tuple[Image, bool]:
        rgb_image = image.convert('RGB').resize(self.settings.input_image_size)

        with torch.no_grad():
            foreground = self._bg_remover.inference(rgb_image.copy())
        foreground_tensor = self.transforms(foreground)
        tensor_rgb = foreground_tensor[:3]
        mask = foreground_tensor[-1]

        return tensor_rgb, mask    
