#!/usr/bin/env python3
"""详情图层分离 ComfyUI 节点。

把仓库根目录的 pipeline.run_pipeline 封装成单个 ComfyUI 节点:
输入一张 IMAGE, 输出 背景层(IMAGE) / 前景元素(IMAGE batch) / 前景蒙版(MASK batch) / manifest(JSON STRING)。
重模型(OCR/rembg/LaMa)沿用 pipeline 的模块级懒加载单例, 跨次执行复用不重载。
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# pipeline.py 已 vendor 进本插件目录, 包内相对导入, 插件自包含。
from . import pipeline

# rembg 支持的抠图模型(前景层)。birefnet 系列质量最好, u2net/isnet 更快更省显存。
_FG_MODELS = ["birefnet-general", "birefnet-massive", "isnet-general-use", "u2net", "u2netp"]


class LayerSeparationNode:
    """详情图层分离: 背景(LaMa inpaint) + 前景元素(BiRefNet 抠图) + 文字(OCR/可选VLM)。"""

    CATEGORY = "image/layer_separation"
    FUNCTION = "separate"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("background", "elements", "element_masks", "manifest")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入的详情图(IMAGE)。"}),
            },
            "optional": {
                "use_vlm": ("BOOLEAN", {"default": True, "label_on": "VLM文字分类开", "label_off": "VLM关"}),
                "fg_model": (_FG_MODELS, {
                    "default": "birefnet-general",
                    "tooltip": "前景抠图模型。birefnet 系列质量最好, u2net/isnet 更快更省显存。",
                }),
                "dashscope_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "DashScope(qwen-vl) 密钥。留空则回退到环境变量 DASHSCOPE_API_KEY 或仓库 .env。use_vlm 关时此项忽略。",
                }),
                "min_area": ("FLOAT", {
                    "default": 0.0015, "min": 0.0, "max": 0.2, "step": 0.0005, "round": 0.0001,
                    "tooltip": "最小元素面积占比。越小保留越多小元素(也更易出噪点)。详情图拆不出多元素时调小。",
                }),
                "close_ksize": ("INT", {
                    "default": 3, "min": 0, "max": 25, "step": 1,
                    "tooltip": "形态学闭运算核大小。0/1=关闭(避免把相邻元素粘成一团); 越大越易把多个元素合并成一个。",
                }),
                "alpha_thr": ("INT", {
                    "default": 30, "min": 0, "max": 254, "step": 1,
                    "tooltip": "前景 alpha 二值化阈值。越高越严格(只留实心主体)。",
                }),
                "ocr_min_score": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "OCR 置信度下限, 低于此的文字丢弃。",
                }),
                "vlm_model": ("STRING", {
                    "default": "qwen-vl-max", "multiline": False,
                    "tooltip": "VLM 模型名(DashScope), 如 qwen-vl-max / qwen-vl-plus。use_vlm 关时忽略。",
                }),
                "element_mode": (["canvas", "cropped"], {
                    "default": "canvas",
                    "tooltip": "canvas=元素贴回原画布(便于直接合成, 4K 多元素内存大); cropped=只输出裁剪小图(省内存, 原位置见 manifest bbox)。",
                }),
            },
        }

    # ---------------- tensor <-> 文件/PIL ----------------
    @staticmethod
    def _frame_to_pil(frame):
        """单帧 IMAGE [H,W,3] float0-1 -> PIL RGB。"""
        arr = (frame.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr, "RGB")

    @staticmethod
    def _pil_to_image_tensor(pil_rgb):
        """PIL RGB -> IMAGE tensor [1,H,W,3] float0-1。"""
        arr = np.asarray(pil_rgb.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr)[None, ...]

    @staticmethod
    def _concat_image_batches(batches):
        """把多帧产生的 [n_i,H_i,W_i(,C)] 批次在 batch 维拼成一个; 尺寸不一时右下补零对齐到最大 H/W。
        支持 IMAGE(4D) 与 MASK(3D)。单批次直接返回, 不做无谓拷贝。"""
        batches = [b for b in batches if b is not None and b.shape[0] > 0]
        if not batches:
            return None
        if len(batches) == 1:
            return batches[0]
        maxH = max(t.shape[1] for t in batches)
        maxW = max(t.shape[2] for t in batches)
        out = []
        for t in batches:
            n, h, w = t.shape[0], t.shape[1], t.shape[2]
            if h != maxH or w != maxW:
                shape = (n, maxH, maxW, t.shape[3]) if t.dim() == 4 else (n, maxH, maxW)
                pad = torch.zeros(shape, dtype=t.dtype)
                pad[:, :h, :w, ...] = t
                t = pad
            out.append(t)
        return torch.cat(out, 0)

    def _composite_elements(self, manifest, workdir, element_mode="canvas"):
        """把 N 个 RGBA 前景小图切成 IMAGE batch + MASK batch。
          canvas : 按 bbox 贴回 (W,H) 全画布, 与 manifest 渲染语义一致, 便于直接合成。
          cropped: 只输出裁剪后的元素小图(统一补零到全体 bbox 最大宽高), 省内存; 原位置见 bbox。
        RGB 在 alpha=0 处清零, 让 PreviewImage 只显示抠图本体。"""
        W = int(manifest["canvas"]["width"])
        H = int(manifest["canvas"]["height"])
        items = [it for it in manifest.get("images", []) if it["bbox"][2] > 0 and it["bbox"][3] > 0]
        if not items:  # N=0 兜底, 给下游一帧全0, 避免空 batch 崩
            return torch.zeros((1, H, W, 3), dtype=torch.float32), torch.zeros((1, H, W), dtype=torch.float32)
        cw = max(it["bbox"][2] for it in items)
        ch = max(it["bbox"][3] for it in items)
        imgs, masks = [], []
        for item in items:
            x, y, w, h = item["bbox"]
            el = Image.open(Path(workdir) / item["url"]).convert("RGBA")
            if el.size != (w, h):
                el = el.resize((w, h), Image.LANCZOS)
            if element_mode == "cropped":
                canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
                canvas.paste(el, (0, 0))
            else:
                canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                canvas.paste(el, (int(x), int(y)))
            rgba = np.asarray(canvas, dtype=np.float32) / 255.0  # [H,W,4]
            alpha = rgba[..., 3]
            rgb = rgba[..., :3] * alpha[..., None]               # 透明处清零
            imgs.append(torch.from_numpy(rgb))
            masks.append(torch.from_numpy(alpha))
        return torch.stack(imgs, 0), torch.stack(masks, 0)

    # ---------------- 主入口 ----------------
    def separate(self, image, use_vlm=True, fg_model="birefnet-general", dashscope_api_key="",
                 min_area=0.0015, close_ksize=3, alpha_thr=30, ocr_min_score=0.5,
                 vlm_model="qwen-vl-max", element_mode="canvas"):
        # 统一密钥/模型入口: 节点上填了就写进环境变量, pipeline 优先读 env;
        # 留空则沿用 env / 仓库 .env 的现有兜底。
        key = (dashscope_api_key or "").strip()
        if use_vlm and key:
            os.environ["DASHSCOPE_API_KEY"] = key
        if use_vlm and (vlm_model or "").strip():
            os.environ["DASHSCOPE_VLM_MODEL"] = vlm_model.strip()

        fg_kwargs = {"min_area": float(min_area), "close_ksize": int(close_ksize), "alpha_thr": int(alpha_thr)}

        # 遍历整个输入 batch, 逐帧分离; B=1 时与单图行为一致。
        bgs, elem_batches, mask_batches, manifests = [], [], [], []
        for b in range(image.shape[0]):
            workdir = tempfile.mkdtemp(prefix="comfy_layersep_")
            input_png = Path(workdir) / "input.png"
            self._frame_to_pil(image[b]).save(input_png)

            manifest = pipeline.run_pipeline(
                str(input_png), workdir, stem="input", use_vlm=use_vlm, fg_model=fg_model,
                fg_kwargs=fg_kwargs, text_min_score=float(ocr_min_score),
            )

            W = int(manifest["canvas"]["width"])
            H = int(manifest["canvas"]["height"])
            bg_pil = Image.open(Path(workdir) / manifest["background"]).convert("RGB")
            if bg_pil.size != (W, H):  # 防御: 背景与 canvas 必须同尺寸
                bg_pil = bg_pil.resize((W, H), Image.LANCZOS)
            bgs.append(self._pil_to_image_tensor(bg_pil))  # [1,H,W,3]
            el, mk = self._composite_elements(manifest, workdir, element_mode=element_mode)
            elem_batches.append(el)
            mask_batches.append(mk)
            manifests.append(manifest)

        background = self._concat_image_batches(bgs)
        elements = self._concat_image_batches(elem_batches)
        element_masks = self._concat_image_batches(mask_batches)
        # 单图输出 manifest 对象(向后兼容); 多图输出 manifest 数组。
        manifest_json = json.dumps(manifests[0] if len(manifests) == 1 else manifests, ensure_ascii=False)
        return (background, elements, element_masks, manifest_json)


class SaveTextNode:
    """把 STRING(如 manifest) 落成 ComfyUI output 目录下的文本文件。

    ComfyUI 的 STRING 连线数据不会被 API 直接回传, 必须由 OUTPUT_NODE 写成 output
    目录里的文件才能被回传/下载。把本节点接在「详情图层分离」的 manifest 输出后面,
    即可拿到 .txt/.json 文件。
    """

    CATEGORY = "image/layer_separation"
    FUNCTION = "save"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "tooltip": "要保存的文本(把 manifest 连进来)。"}),
                "filename_prefix": ("STRING", {"default": "manifest", "tooltip": "文件名前缀, 自动追加递增序号。可含子目录, 如 layersep/manifest。"}),
                "extension": (["txt", "json"], {"default": "txt", "tooltip": "文件扩展名。"}),
            },
        }

    def save(self, text, filename_prefix="manifest", extension="txt"):
        # 延迟导入: ComfyUI 运行时才有 folder_paths, 包外导入本模块不受影响。
        import folder_paths

        out_dir = folder_paths.get_output_directory()
        full_dir, fname, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        filename = f"{fname}_{counter:05d}.{extension}"
        path = os.path.join(full_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text if isinstance(text, str) else str(text))
        # ui.text 给前端预览; 文件已落 output, 可被 ComfyUI API 回传/下载。
        return {"ui": {"text": [text], "string": [text],
                       "files": [{"filename": filename, "subfolder": subfolder, "type": "output"}]}}


NODE_CLASS_MAPPINGS = {
    "LayerSeparation": LayerSeparationNode,
    "LayerSeparationSaveText": SaveTextNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerSeparation": "详情图层分离 (Layer Separation)",
    "LayerSeparationSaveText": "保存文本 (Save Text)",
}
