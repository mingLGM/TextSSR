from typing import List, Optional, Union, Dict, Tuple
import PIL
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from diffusers import AutoencoderKL, DiffusionPipeline, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from anyword_data import AnyWordDataset # 自定义数据集类
import os
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import torchvision.transforms as transforms
import random
import cv2

# -----------------------------
# 自定义扩散模型推理 Pipeline
# -----------------------------
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
        # VAE 下采样因子
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
            guidance_scale (float): 引导系数，默认为 7.5。
            device (torch.device): 使用的计算设备。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 解码后的图像张量和未解码的潜在表示。
        """
        if mask_image is None:
            raise ValueError("`mask_image` input cannot be undefined.")

        batch_size = prompt.shape[0]
        vae.to(device)
        unet.to(device)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Preprocess mask and image. 缩放并预处理遮罩和图像
        vae_scale_factor = self.vae_scale_factor
        _, _, mask_height, mask_width = mask.size()
        mask = torch.nn.functional.interpolate(mask, size=[mask_width // vae_scale_factor, mask_height // vae_scale_factor])

        # 编码 glyph 和 masked image 到 latent 空间
        glyph_latents = vae.encode(glyph).latent_dist.sample() * vae.config.scaling_factor
        masked_image_latents = vae.encode(mask_image).latent_dist.sample() * vae.config.scaling_factor
        
        do_classifier_free_guidance = guidance_scale > 1.0
        # Prepare prompt embeddings
        if do_classifier_free_guidance:
            # Negative prompt: zero tensor (or learnable null embedding)
            uncond_prompt = torch.zeros_like(prompt)
            prompt = torch.cat([uncond_prompt, prompt])  # [2B, C, H, W]
            # Duplicate latents for unconditional + conditional
            glyph_latents = torch.cat([glyph_latents, glyph_latents])
            masked_image_latents = torch.cat([masked_image_latents, masked_image_latents])
            mask = torch.cat([mask, mask])

        # 初始化随机噪声 latent
        shape = (batch_size, vae.config.latent_channels, mask_height // vae_scale_factor, mask_width // vae_scale_factor)
        latents = randn_tensor(shape, generator=torch.manual_seed(20), device=device) * self.scheduler.init_noise_sigma

        # 扩散步骤
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for t in timesteps:
                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latent_model_input, latent_model_input])
                
                # glyph_latents. 将多个模态特征拼接作为 UNet 输入, 拼接所有条件：noisy latent + masked image latent + glyph latent + mask
                sample = torch.cat([latent_model_input, masked_image_latents, glyph_latents, mask], dim=1)
                noise_pred = unet(sample=sample, timestep=t, encoder_hidden_states=prompt, ).sample
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                progress_bar.update()
        
        # 解码生成图像
        pred_latents = latents / vae.config.scaling_factor
        image_vae = vae.decode(pred_latents).sample
        image = (image_vae / 2 + 0.5) * 255.0
        return image, image_vae


def regenerate_conditions_for_text(text, rect, dataset, ttf_size=64, max_len=25, resolution=256, font_path="AlibabaPuHuiTi-3-85-Bold.ttf"):
    """为给定文本重新生成TTF和glyph条件"""
    # 使用数据集的draw_text方法生成TTF
    ttf_img = dataset.draw_text(text[:max_len], font_path)
    
    # 使用数据集的draw_glyph方法生成glyph
    glyph_img = dataset.draw_glyph(text, rect, font_path)
    
    return ttf_img, glyph_img

# 辅助函数：获取rect（与generate_mask一致）
def get_rect_from_mask(mask_tensor):
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
    center = (128, 128)
    size = (100, 50)
    return (center, size, 0.0)

# 文本变体生成
def generate_text_variations(text, num_variations=3):
    """为文本生成多种顺序变体"""
    if len(text) <= 1:
        return [text] * num_variations
    
    variations = [text]  # 原始
    
    if len(text) > 1:
        variations.append(text[::-1])  # 反转
        
        # 随机
        for _ in range(max(0, num_variations - 2)):
            char_list = list(text)
            random.shuffle(char_list)
            variations.append(''.join(char_list))
    
    return variations[:num_variations]



# -----------------------------
# 模型加载, Initialize models.
# -----------------------------
vae = AutoencoderKL.from_pretrained("./model/vae_ft/checkpoint-150000/vae", use_safetensors=False, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained("./model/step2/checkpoint-25000/unet", use_safetensors=False)
# unet = UNet2DConditionModel.from_pretrained("stability-ai/stable-diffusion-2-1-base", subfolder="unet")
noise_scheduler = DDPMScheduler.from_pretrained("./model/stable-diffusion-v2-1/scheduler")
# noise_scheduler = DDIMScheduler.from_pretrained("./model/stable-diffusion-v2-1/scheduler")


# Create pipeline. 创建推理管道实例
pipe = StableDiffusionPipeline(vae=vae, unet=unet, scheduler=noise_scheduler)

resolution = 512          # 输入图像分辨率
num_inference_steps = 75  # 推理步数
guidance_scale = 4.0      #  引导系数

# -----------------------------
# 输出目录设置
# -----------------------------
save_dir = "./output/gangban_t"
os.makedirs(os.path.join(save_dir, "region"),  exist_ok=True)
os.makedirs(os.path.join(save_dir, "local"),  exist_ok=True)

# Create dataset and dataloader. 加载测试数据集
datasets = AnyWordDataset(
    json_path="./benchmark/gangban/test_t.json",  # 测试集标注文件
    resolution=resolution,                      # 输入图像分辨率
    seed=42,
    ttf_size=64,                             # 渲染 TTF 字体大小
    max_len=25,                              # 最大文本长度
)

# Set batch size to 1 for single card inference. 构建数据加载器，设置批大小为32以加速推理
dataloader = DataLoader(datasets, shuffle=False, batch_size=1, num_workers=16)

cnt = 0
results = {}
MAX_SAMPLES = 100
FONT_PATH = "AlibabaPuHuiTi-3-85-Bold.ttf"
VARIANTS_PER_SAMPLE = 3  # 每个样本生成3种顺序变体

for batch_idx, batch in enumerate(tqdm(dataloader)):
    if cnt >= MAX_SAMPLES:
        break
    
    imgs = batch["image"].to("cuda")                  # 原始高清图（可能未使用）
    masked_images = batch["masked_image"].to("cuda")  # 被 mask 的低清图
    masks = batch["mask"].to("cuda")                  # mask (1=保留, 0=修复区域)
    # ttf_imgs = batch["ttf_img"].to("cuda")            # TTF 渲染的文本图像（作为 prompt）
    # glyphs = batch["glyph"].to("cuda")                # 文字骨架
    # texts = batch["text"]                             # 对应文本字符串
    original_texts = batch["text"]                             # 对应文本字符串
    
    
    # 处理每个样本
    for i, orig_text in enumerate(original_texts):
        if cnt >= MAX_SAMPLES:
            break
            
        print(f"\n处理文本: {orig_text}")
        
        # 生成文本变体
        text_variations = generate_text_variations(orig_text, VARIANTS_PER_SAMPLE)
        
        for var_idx, variant_text in enumerate(text_variations):
            if variant_text == orig_text and var_idx > 0:
                continue
                
            print(f"  生成变体 {var_idx+1}: {variant_text}")
            
            # 获取rect（确保与原始一致）
            rect = get_rect_from_mask(masks[i])
            
            # 重新生成条件
            ttf_imgs, glyph_imgs = regenerate_conditions_for_text(
                variant_text, 
                rect, 
                datasets,  # 传入数据集实例以使用其方法
                ttf_size=64,
                max_len=25,
                resolution=resolution,
                font_path=FONT_PATH
            )
            
            # 移到GPU
            ttf_imgs = ttf_imgs.unsqueeze(0).to("cuda")
            glyph_imgs = glyph_imgs.unsqueeze(0).to("cuda")
            
            # 当前样本的其他数据
            current_masked_image = masked_images[i:i+1]
            current_mask = masks[i:i+1]
            
            try:
                # 图像生成主调用
                image, image_vae = pipe(
                    prompt=ttf_imgs,
                    glyph=glyph_imgs,
                    mask_image=current_masked_image,
                    mask=current_mask,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    device=torch.device("cuda")
                )
                
                # # 将 image_vae 转换为 [0, 255] 范围，并调整维度顺序以适应 PIL 图像格式
                # image_vae_rescaled = ((image_vae / 2 + 0.5) * 255).clamp(0, 255).to(torch.uint8)
                # image_vae_rescaled = image_vae_rescaled.permute(0, 2, 3, 1)  # 移动通道维度到最后一维
                # for i, img in enumerate(image_vae_rescaled):
                #     # 创建 PIL Image 对象并保存
                #     img_pil = Image.fromarray(img.cpu().numpy(), 'RGB')
                #     file_idx = i + cnt
                #     img_pil.save(os.path.join(save_dir, "local", f"{file_idx}_vae.png"))  # 确保目录存在
                
                # 处理每一张生成结果，并保存局部裁剪图及完整图
                for img_idx, img in enumerate(image):
                    
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
                    
                    # 保存完整图像
                    file_idx = cnt
                    img_np = img.cpu().permute(1, 2, 0).float().detach().numpy().astype(np.uint8)
                    output_filename = f"{file_idx}_{variant_type}_{variant_text}.png"
                    Image.fromarray(img_np).save(os.path.join(save_dir, "local", output_filename))
                    
                    # 保存裁剪区域图像
                    cropped_output_image_np = cropped_output_image.cpu().permute(1, 2, 0).float().detach().numpy().astype(np.uint8)
                    cropped_output_image_pil = Image.fromarray(cropped_output_image_np)
                    file_path = os.path.join(save_dir, "region", output_filename)
                    cropped_output_image_pil.save(file_path)
                    
                    # 记录结果
                    results[output_filename] = {
                        "original_text": orig_text,
                        "generated_text": variant_text,
                        "variant_type": variant_type
                    }
                    
                    cnt += 1
                    print(f"     已保存: {output_filename}")
                    
            except Exception as e:
                print(f"     生成失败: {e}")
                continue


# Save results. 最终将标签信息写入 JSON 文件
with open(f"{save_dir}/labels.json", 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)
