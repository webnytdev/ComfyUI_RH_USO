import os
import dataclasses
from typing import Literal

from accelerate import Accelerator
from transformers import HfArgumentParser
from PIL import Image
import json
import itertools
import torch

from .uso.flux.pipeline import USOPipeline, preprocess_ref
from transformers import SiglipVisionModel, SiglipImageProcessor
from tqdm import tqdm
import folder_paths
import numpy as np
import comfy.utils

class RH_USO_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                
            },
        }

    RETURN_TYPES = ("RHUSOMudules",)
    RETURN_NAMES = ("USO Modules",)
    FUNCTION = "load"

    CATEGORY = "Runninghub/USO"

    def load(self, **kwargs):
        # accelerator = Accelerator()
        device = 'cuda'
        siglip_path = os.path.join(folder_paths.models_dir, 'clip', 'siglip-so400m-patch14-384')
        siglip_processor = SiglipImageProcessor.from_pretrained(
            siglip_path
        )
        siglip_model = SiglipVisionModel.from_pretrained(
            siglip_path
        )
        siglip_model.eval()
        siglip_model.to(device)
        print("SigLIP model loaded successfully")

        # hardcode hyperparamters -kiki
        model_type = 'flux-dev-fp8'
        lora_rank = 128

        pipeline = USOPipeline(
            model_type,
            device,
            True, #args.offload,
            only_lora=True,
            lora_rank=lora_rank,
            hf_download=False,
        )
        if siglip_model is not None:
            pipeline.model.vision_encoder = siglip_model
            print('-----> hook siglip encoder')
            return ({'siglip_processor':siglip_processor, 'pipeline':pipeline}, )

class RH_USO_Sampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "uso": ("RHUSOMudules", ),
                "prompt": ("STRING", {"multiline": True,
                                      'default': ''}),
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
                "num_inference_steps": ("INT", {"default": 25}),
                "guidance": ("FLOAT", {"default": 4.0}),
                "seed": ("INT", {"default": 20, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
            },
            "optional": {
                "content_image": ("IMAGE", ),
                "style_image": ("IMAGE", ),
                "style2_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"

    CATEGORY = "Runninghub/USO"

    def tensor_2_pil(self, img_tensor):
        if img_tensor is not None:
            i = 255. * img_tensor.squeeze().cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            return img
        else:
            return None
        
    def preprocess_ref(self, raw_image: Image.Image, long_size: int = 512, scale_ratio: int = 1):
        # 获取原始图像的宽度和高度
        image_w, image_h = raw_image.size
        if image_w == image_h and image_w == 16:
            return raw_image

        # 计算长边和短边
        if image_w >= image_h:
            new_w = long_size
            new_h = int((long_size / image_w) * image_h)
        else:
            new_h = long_size
            new_w = int((long_size / image_h) * image_w)

        # 按新的宽高进行等比例缩放
        raw_image = raw_image.resize((new_w, new_h), resample=Image.LANCZOS)

        # 为了能让canny img进行scale
        scale_ratio = int(scale_ratio)
        target_w = new_w // (16 * scale_ratio) * (16 * scale_ratio)
        target_h = new_h // (16 * scale_ratio) * (16 * scale_ratio)

        # 计算裁剪的起始坐标以实现中心裁剪
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h

        # 进行中心裁剪
        raw_image = raw_image.crop((left, top, right, bottom))

        # 转换为 RGB 模式
        raw_image = raw_image.convert("RGB")
        return raw_image

    def sample(self, **kwargs):
        ref_imgs = []
        content_image = self.tensor_2_pil(kwargs.get('content_image', None))
        style_image = self.tensor_2_pil(kwargs.get('style_image', None))
        style2_image = self.tensor_2_pil(kwargs.get('style2_image', None))
        print(f'conds-c/s1/s2:{content_image is not None} {style_image is not None} {style2_image is not None}')
        ref_imgs.append(content_image)
        if style_image is not None:
            ref_imgs.append(style_image)
        if style2_image is not None:
            ref_imgs.append(style2_image)
        siglip_inputs = None

        width = kwargs.get('width')
        height = kwargs.get('height')
        prompt = kwargs.get('prompt')
        guidance = kwargs.get('guidance')
        num_steps = kwargs.get('num_inference_steps')
        seed = kwargs.get('seed') % (2 ** 32)

        # hardcode hyperparameters -kiki
        content_ref = 512
        pe = 'd'

        uso = kwargs.get('uso')
        siglip_processor = uso['siglip_processor']
        pipeline = uso['pipeline']
        with torch.no_grad():
            siglip_inputs = [
                siglip_processor(img, return_tensors="pt").to(pipeline.device) 
                for img in ref_imgs[1:] if isinstance(img, Image.Image)
                ]

        ref_imgs_pil = [
            self.preprocess_ref(img, content_ref) for img in ref_imgs[:1] if isinstance(img, Image.Image)
        ]
        self.pbar = comfy.utils.ProgressBar(num_steps)

        image_gen = pipeline(
            prompt=prompt,
            width=width,
            height=height,
            guidance=guidance,
            num_steps=num_steps,
            seed=seed,
            ref_imgs=ref_imgs_pil,
            pe=pe,
            siglip_inputs=siglip_inputs,
            update_func=self.update,
        )

        image = np.array(image_gen).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image, )
    
    def update(self):
        self.pbar.update(1)
        

NODE_CLASS_MAPPINGS = {
    "RunningHub USO Loader": RH_USO_Loader,
    "RunningHub USO Sampler":RH_USO_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHub USO Loader": "RunningHub USO Loader",
    "RunningHub USO Sampler": "RunningHub USO Sampler",
} 