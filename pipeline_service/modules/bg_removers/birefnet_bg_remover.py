import gc
from PIL import Image

import torch
from torchvision import transforms
from modules.bg_removers.base_bg_remover import BaseBGRemover
from modules.bg_removers.birefnet_model.birefnet import BiRefNet
from config.settings import BackgroundRemovalConfig


class BiRefNetBGRemover(BaseBGRemover):
    def __init__(self, settings: BackgroundRemovalConfig):
        super().__init__(settings)
        self._bg_remover: BiRefNet | None = None
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        self._transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.settings.input_image_size), 
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )

    def load_model(self) -> None:
        self._bg_remover = BiRefNet.from_pretrained('ZhengPeng7/BiRefNet_dynamic')
        self._bg_remover.to(self.device)
        self._bg_remover.eval()
        self._bg_remover.half()

    def unload_model(self) -> None:
        del self._bg_remover
        gc.collect()
        torch.cuda.empty_cache()
        self._bg_remover = None

    def remove_bg(self, image: Image) -> tuple[Image, bool]:
        result_image = image.convert('RGB').resize(self.settings.input_image_size)

        input_image = self._transform_image(result_image).unsqueeze(0).to(self.device).half()
        with torch.no_grad():
            preds = self._bg_remover(input_image)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(result_image.size)
        result_image.putalpha(mask)

        foreground_tensor = self.transforms(result_image)
        tensor_rgb = foreground_tensor[:3]
        mask = foreground_tensor[-1]

        return tensor_rgb, mask    
