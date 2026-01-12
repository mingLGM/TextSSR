import warnings, torch
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
from typing import List, Optional, Union, Dict, Tuple
import PIL
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from diffusers import AutoencoderKL, DiffusionPipeline, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
import os
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import torchvision.transforms as transforms
import random
import cv2
from anyword_data import AnyWordDataset  # 自定义数据集类

class TextSSRGenerator:
    """
    TextSSR生成器类，封装文本顺序变体生成功能
    
    参数:
        model_dir: 模型目录路径
        resolution: 图像分辨率，默认512
        ttf_size: TTF字体大小，默认64
        max_len: 最大文本长度，默认25
        font_path: 字体文件路径
        device: 设备，默认"cuda"
    """
    
    def __init__(
        self,
        model_dir: str = "./model",
        resolution: int = 256,
        ttf_size: int = 64,
        max_len: int = 25,
        font_path: str = "AlibabaPuHuiTi-3-85-Bold.ttf",
        device: str = "cuda",
        datasets: AnyWordDataset = None,
    ):
        self.model_dir = model_dir
        self.resolution = resolution
        self.ttf_size = ttf_size
        self.max_len = max_len
        self.font_path = font_path
        self.device = device
        self.datasets = datasets
        
        # 加载模型
        self._load_models()
        
        # 创建pipeline
        self.pipe = self._create_pipeline()
        
        print(f"TextSSRGenerator初始化完成，设备: {device}")
    
    def _load_models(self):
        """加载模型"""
        print("正在加载模型...")
        
        try:
            # VAE
            vae_path = os.path.join(self.model_dir, "vae_ft/checkpoint-150000/vae")
            self.vae = AutoencoderKL.from_pretrained(
                vae_path, 
                use_safetensors=False, 
                subfolder="vae"
            )
            print(f"✓ VAE加载完成: {vae_path}")
        except Exception as e:
            print(f"✗ VAE加载失败: {e}")
            raise
        
        try:
            # UNet
            unet_path = os.path.join(self.model_dir, "step2/checkpoint-25000/unet")
            # unet_path = os.path.join(self.model_dir, "stability-ai/stable-diffusion-2-1-base")
            self.unet = UNet2DConditionModel.from_pretrained(
                unet_path, 
                use_safetensors=False,
                subfolder="unet"
            )
            print(f"✓ UNet加载完成: {unet_path}")
        except Exception as e:
            print(f"✗ UNet加载失败: {e}")
            raise
        
        try:
            # Scheduler
            scheduler_path = os.path.join(self.model_dir, "stable-diffusion-v2-1/scheduler")
            self.scheduler = DDPMScheduler.from_pretrained(scheduler_path)
            # self.scheduler = DDIMScheduler.from_pretrained(scheduler_path)
            print(f"✓ Scheduler加载完成: {scheduler_path}")
        except Exception as e:
            print(f"✗ Scheduler加载失败: {e}")
            raise
    
    def _create_pipeline(self) -> DiffusionPipeline:
        """创建推理管道"""
        
        class StableDiffusionPipeline(DiffusionPipeline):
            """
            自定义的 Stable Diffusion 推理 Pipeline 类，用于基于条件输入（如文本、字形图像等）进行图像生成。
            
            参数:
                vae (AutoencoderKL): 变分自编码器，用于图像与潜在空间之间的转换。
                unet (UNet2DConditionModel): 条件 UNet 模型，负责噪声预测。
                scheduler (DDPMScheduler): 噪声调度器，控制扩散过程的时间步长。
            """
            def __init__(self, vae: AutoencoderKL, unet: UNet2DConditionModel, scheduler: DDPMScheduler):
                super().__init__()
                self.register_modules(vae=vae, unet=unet, scheduler=scheduler)
                self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            
            @property
            def _execution_device(self):
                """
                获取当前模型执行设备。如果使用了 Hook，则从 Hook 中获取实际运行设备。

                返回:
                    torch.device: 当前模型执行设备。
                """
                if not hasattr(self.unet, "_hf_hook"):
                    return self.device
                for module in self.unet.modules():
                    if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "execution_device"):
                        return torch.device(module._hf_hook.execution_device)
                return self.device
            
            @torch.no_grad()
            def __call__(
                self,
                prompt: Union[torch.FloatTensor, PIL.Image.Image],
                glyph: Union[torch.FloatTensor, PIL.Image.Image],
                mask_image: Union[torch.FloatTensor, PIL.Image.Image],
                mask: Union[torch.FloatTensor, PIL.Image.Image],
                num_inference_steps: int = 50,
                guidance_scale: float = 5.0,
                device=None
            ):
                """
                执行一次完整的图像生成流程。

                参数:
                    prompt (Union[torch.FloatTensor, PIL.Image.Image]): TTF 渲染的文本图像(作为 condition),文本或图像形式的提示信息。
                    glyph (Union[torch.FloatTensor, PIL.Image.Image]): 字形图像输入(文字骨架图)。
                    mask_image (Union[torch.FloatTensor, PIL.Image.Image]): 被遮罩的原始图像。
                    mask (Union[torch.FloatTensor, PIL.Image.Image]): 遮罩图像。
                    num_inference_steps (int): 推理迭代次数，默认为 50。
                    guidance_scale (float): 引导系数，默认为 3.0。
                    device (torch.device): 使用的计算设备。

                返回:
                    Tuple[torch.Tensor, torch.Tensor]: 解码后的图像张量和未解码的潜在表示。
                """
                if mask_image is None:
                    raise ValueError("`mask_image` input cannot be undefined.")

                batch_size = prompt.shape[0]
                self.vae.to(device)
                self.unet.to(device)
                self.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = self.scheduler.timesteps

                # Preprocess mask and image. 缩放并预处理mask和图像
                vae_scale_factor = self.vae_scale_factor
                _, _, mask_height, mask_width = mask.size()
                mask = torch.nn.functional.interpolate(
                    mask, 
                    size=[mask_width // vae_scale_factor, mask_height // vae_scale_factor]
                )

                # 编码glyph和masked image到潜在空间
                glyph_latents = self.vae.encode(glyph).latent_dist.sample() * self.vae.config.scaling_factor
                masked_image_latents = self.vae.encode(mask_image).latent_dist.sample() * self.vae.config.scaling_factor
                
                do_classifier_free_guidance = guidance_scale > 1.0
                
                # 准备prompt embeddings
                if do_classifier_free_guidance:
                    # Negative prompt: zero tensor (or learnable null embedding)
                    uncond_prompt = torch.zeros_like(prompt)
                    prompt = torch.cat([uncond_prompt, prompt])  # [2B, C, H, W]
                    # Duplicate latents for unconditional + conditional
                    glyph_latents = torch.cat([glyph_latents, glyph_latents])
                    masked_image_latents = torch.cat([masked_image_latents, masked_image_latents])
                    mask = torch.cat([mask, mask])

                # 初始化随机噪声
                shape = (
                    batch_size, 
                    self.vae.config.latent_channels, 
                    mask_height // vae_scale_factor, 
                    mask_width // vae_scale_factor
                )
                latents = randn_tensor(
                    shape, 
                    generator=torch.manual_seed(20), 
                    device=device
                ) * self.scheduler.init_noise_sigma

                # 扩散步骤
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    for t in timesteps:
                        latent_model_input = latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                        
                        if do_classifier_free_guidance:
                            latent_model_input = torch.cat([latent_model_input, latent_model_input])
                        
                        # 拼接所有条件
                        # glyph_latents. 将多个模态特征拼接作为 UNet 输入, 拼接所有条件：noisy latent + masked image latent + glyph latent + mask
                        sample = torch.cat([latent_model_input, masked_image_latents, glyph_latents, mask], dim=1)
                        noise_pred = self.unet(
                            sample=sample, 
                            timestep=t, 
                            encoder_hidden_states=prompt
                        ).sample
                        
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        
                        latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                        progress_bar.update()
                
                # 解码生成图像
                pred_latents = latents / self.vae.config.scaling_factor
                image_vae = self.vae.decode(pred_latents).sample
                image = (image_vae / 2 + 0.5) * 255.0
                return image, image_vae
        
        return StableDiffusionPipeline(self.vae, self.unet, self.scheduler)
    
    def _get_rect_from_mask(self, mask_tensor: torch.Tensor) -> tuple:
        """从mask tensor获取rect，与generate_mask一致"""
        mask_np = mask_tensor.squeeze().cpu().numpy()
        mask_np = np.transpose(mask_np)  # 重要：与generate_mask保持一致！
        points = np.column_stack(np.where(mask_np == 0))  # 注意：mask中0是待修复区域
        
        if len(points) >= 5:
            try:
                points = np.float32(points)
                rect = cv2.minAreaRect(points)
                return rect
            except Exception as e:
                print(f"计算rect失败: {e}")
        
        # 默认rect
        center = (self.resolution // 2, self.resolution // 2)
        size = (self.resolution // 4, self.resolution // 8)
        return (center, size, 0.0)
    
    def generate_text_variations(
        self, 
        text: str, 
        num_variations: int = 3,
        variation_types: List[str] = None
    ) -> List[str]:
        """
        生成文本顺序变体
        
        参数:
            text: 原始文本
            num_variations: 变体数量
            variation_types: 变体类型列表，可选["original", "reverse", "random"]
        
        返回:
            文本变体列表
        """
        if variation_types is None:
            variation_types = ["original", "reverse", "random"]
        
        if len(text) <= 1:
            return [text] * min(num_variations, len(variation_types))
        
        variations = []
        
        for v_type in variation_types[:num_variations]:
            if v_type == "original":  # 原始字符串
                variations.append(text)
            elif v_type == "reverse" and len(text) > 1:  # 反转字符串
                variations.append(text[::-1])
            elif v_type == "random" and len(text) > 1:  # 随机字符串
                char_list = list(text)
                random.shuffle(char_list)
                variations.append(''.join(char_list))
            else:
                variations.append(text)
        
        return variations[:num_variations]
    
    def regenerate_conditions(
        self, 
        text: str, 
        rect: tuple, 
        font_path: str = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为给定文本重新生成TTF和glyph条件
        
        参数:
            text: 文本
            rect: 矩形参数
            font_path: 字体路径
        
        返回:
            (ttf_img, glyph_img) 条件张量
        """
        if font_path is None:
            font_path = self.font_path
        ttf_img = self.datasets.draw_text(text[:self.max_len], font_path)
        glyph_img = self.datasets.draw_glyph(text, rect, font_path)
        
        return ttf_img, glyph_img
    
    def generate_for_batch(
        self,
        batch: Dict[str, torch.Tensor],
        num_variations: int = 3,
        num_inference_steps: int = 75,
        guidance_scale: float = 4.0,
        save_dir: str = "./output",
        save_cropped: bool = True,
        progress_bar: bool = True, 
        cnt: int = 0,
    ) -> Dict[str, Dict]:
        """
        为批次数据生成文本变体图像
        
        参数:
            batch: 数据批次，包含以下键:
                - "masked_image": 被mask的低清图
                - "mask": mask (1=保留, 0=修复区域)
                - "text": 原始文本
            num_variations: 每个样本生成多少种变体
            num_inference_steps: 推理步数
            guidance_scale: 引导系数
            save_dir: 保存目录
            save_cropped: 是否保存裁剪区域
            progress_bar: 是否显示进度条
        
        返回:
            结果字典，包含生成信息和文件路径
        """
        # 创建保存目录
        os.makedirs(os.path.join(save_dir, "region"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "local"), exist_ok=True)
        
        results = {}
        imgs = batch.get("image")
        masked_images = batch["masked_image"].to(self.device)
        masks = batch["mask"].to(self.device)
        original_texts = batch["text"]
        file_name = os.path.splitext(batch["file_name"][0])[0]
        
        # 处理每个样本
        iterator = range(len(original_texts))
        if progress_bar:
            iterator = tqdm(iterator, desc=f"{file_name}: 生成文本变体")
        
        for i in iterator:
            orig_text = original_texts[i]
            
            # 生成文本变体
            text_variations = self.generate_text_variations(orig_text, num_variations)
            
            for var_idx, variant_text in enumerate(text_variations):
                if variant_text == orig_text and var_idx > 0:
                    continue
                
                print(f"  第{cnt}个文本 {orig_text}  生成变体 {var_idx+1}: {variant_text}")
                
                # 获取rect
                rect = self._get_rect_from_mask(masks[i])
                
                # 重新生成条件
                ttf_variant, glyph_variant = self.regenerate_conditions(variant_text, rect)
                
                # 移到GPU
                ttf_variant = ttf_variant.unsqueeze(0).to(self.device)
                glyph_variant = glyph_variant.unsqueeze(0).to(self.device)
                
                # 当前样本的其他数据
                current_masked_image = masked_images[i:i+1]
                current_mask = masks[i:i+1]
                
                try:
                    # 图像生成主调用
                    image, image_vae = self.pipe(
                        prompt=ttf_variant,
                        glyph=glyph_variant,
                        mask_image=current_masked_image,
                        mask=current_mask,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        device=torch.device(self.device)
                    )
                    
                    # # 将 image_vae 转换为 [0, 255] 范围，并调整维度顺序以适应 PIL 图像格式
                    # image_vae_rescaled = ((image_vae / 2 + 0.5) * 255).clamp(0, 255).to(torch.uint8)
                    # image_vae_rescaled = image_vae_rescaled.permute(0, 2, 3, 1)  # 移动通道维度到最后一维
                    # for i, img in enumerate(image_vae_rescaled):
                    #     # 创建 PIL Image 对象并保存
                    #     img_pil = Image.fromarray(img.cpu().numpy(), 'RGB')
                    #     file_idx = i + cnt
                    #     img_pil.save(os.path.join(save_dir, "local", f"{file_idx}_vae.png"))  # 确保目录存在
                
                    
                    # 处理生成结果
                    for img_idx, img in enumerate(image):
                        # 获取掩码区域
                        mask_np = current_mask[i].cpu().detach().numpy().astype(np.uint8)
                        coords = np.column_stack(np.where(mask_np == 0))
                        
                        if coords.size > 0:
                            y_min, x_min = coords[:, 1].min(), coords[:, 2].min()
                            y_max, x_max = coords[:, 1].max(), coords[:, 2].max()
                            cropped_output_image = img[:, y_min:y_max+1, x_min:x_max+1]
                        else:
                            cropped_output_image = img
                        
                        # 创建变体标识
                        variant_type = "original" if variant_text == orig_text else \
                                     "reverse" if variant_text == orig_text[::-1] else "random"
                        
                        # 生成文件名
                        filename = f"{file_name}_{cnt}_{var_idx}{variant_type}_{variant_text}.png"
                        
                        # 保存完整图像
                        img_np = img.cpu().permute(1, 2, 0).float().detach().numpy().astype(np.uint8)
                        Image.fromarray(img_np).save(os.path.join(save_dir, "local", filename))
                        
                        # 保存裁剪区域图像
                        if save_cropped and coords.size > 0:
                            cropped_np = cropped_output_image.cpu().permute(1, 2, 0).float().detach().numpy().astype(np.uint8)
                            Image.fromarray(cropped_np).save(os.path.join(save_dir, "region", filename))
                        
                        # 记录结果
                        results[filename] = {
                            "original_text": orig_text,
                            "generated_text": variant_text,
                            "variant_type": variant_type,
                            "full_image_path": os.path.join(save_dir, "local", filename),
                            "cropped_image_path": os.path.join(save_dir, "region", filename) if save_cropped else None
                        }
                        
                        print(f"     已保存: {filename}")
                        
                except Exception as e:
                    print(f"生成图像失败 (文本: '{variant_text}'): {e}")
                    continue
        
        return results
    
    def save_results(self, results: Dict, save_path: str):
        """保存结果到JSON文件"""
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"结果已保存到: {save_path}")


# ====================== 使用示例 ======================

if __name__ == "__main__":
    
    resolution = 512          # 输入图像分辨率
    num_inference_steps = 75  # 推理步数  75
    guidance_scale = 4.0      #  引导系数
    json_path="./benchmark/gangban/test.json"
    save_dir = "./output/gangban"
    
    
    # 示例1: 基本使用
    def example_basic():
        """基本使用示例"""
        
        # 加载数据集
        datasets = AnyWordDataset(
            json_path=json_path,  # 测试集标注文件
            resolution=resolution,                        # 输入图像分辨率
            seed=42,
            ttf_size=64,                                  # 渲染 TTF 字体大小
            max_len=25,                                   # 最大文本长度
        )
        
        # 初始化生成器
        generator = TextSSRGenerator(
            model_dir="./model",
            resolution=resolution,
            font_path="AlibabaPuHuiTi-3-85-Bold.ttf",
            device="cuda",
            datasets = datasets
        )
        
        dataloader = DataLoader(datasets, shuffle=False, batch_size=1, num_workers=16)
        
        # 生成结果
        all_results = {}
        cnt = 0
        MAX_SAMPLES = 100
        
        for batch_idx, batch in enumerate(dataloader):
            if cnt >= MAX_SAMPLES:
                break
            
            # 为当前批次生成文本变体图像
            results = generator.generate_for_batch(
                batch=batch,
                num_variations=3,  # 每个样本生成3种变体
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                save_dir=save_dir,
                save_cropped=True,
                progress_bar=True,
                cnt=cnt,
            )
            
            all_results.update(results)
            cnt += len(batch["text"])

        
        # 保存所有结果
        generator.save_results(all_results, os.path.join(save_dir, "labels.json"))
    
    # # 示例2: 自定义变体类型
    # def example_custom_variations():
    #     """自定义变体类型示例"""
    #     generator = TextSSRGenerator(
    #         model_dir="./model",
    #         resolution=512,
    #         device="cuda"
    #     )
        
    #     # 自定义文本和变体
    #     texts = ["HelloWorld", "123456", "ABCDEF"]
        
    #     # 生成不同变体
    #     for text in texts:
    #         variations = generator.generate_text_variations(
    #             text=text,
    #             num_variations=5,
    #             variation_types=["original", "reverse", "random", "random", "random"]
    #         )
    #         print(f"文本 '{text}' 的变体:")
    #         for i, var in enumerate(variations):
    #             print(f"  {i+1}. {var}")
    
    
    
    
    example_basic()
    
    