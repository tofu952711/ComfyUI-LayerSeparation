#!/usr/bin/env python3
"""详情图层分离 pipeline（可导入模块）。
对外主函数: run_pipeline(image_path, workdir, stem) -> manifest dict
  manifest = {canvas, background, images:[{url,bbox}], texts:[...]}  (url 为 workdir 相对路径)
重模型(OCR/rembg/LaMa)用模块级懒加载单例, 服务里复用不重载。
"""
import base64, json, os, re
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
SIZE_K = 0.62
os.environ.setdefault("U2NET_HOME", str(Path.home() / ".u2net"))

# ---------------- 懒加载单例 ----------------
_ocr = None
_rembg_sessions = {}
_lama = {}          # device -> model
_device = None


def _lama_utils():
    """兼容新旧 simple-lama: 函数从 .utils.util(新) 或 .utils(旧) 导。"""
    try:
        from simple_lama_inpainting.utils.util import prepare_img_and_mask, download_model
    except ImportError:
        from simple_lama_inpainting.utils import prepare_img_and_mask, download_model
    return prepare_img_and_mask, download_model


def get_device():
    """torch 设备: CUDA(Linux GPU) > MPS(Apple GPU) > CPU。"""
    global _device
    if _device is None:
        import torch
        if torch.cuda.is_available():
            _device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"
    return _device


def _onnx_providers():
    """onnxruntime provider 优先级: CUDA > CoreML(Apple GPU) > CPU。"""
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
    except Exception:
        return None
    for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider"):
        if p in avail:
            return [p, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def get_ocr():
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    return _ocr


def get_rembg(model):
    if model not in _rembg_sessions:
        from rembg import new_session
        prov = _onnx_providers()
        try:
            _rembg_sessions[model] = new_session(model, providers=prov) if prov else new_session(model)
        except Exception:
            _rembg_sessions[model] = new_session(model)   # provider 不行回退默认
    return _rembg_sessions[model]


def get_lama(device):
    if device not in _lama:
        import torch
        _, download_model = _lama_utils()
        mp = download_model("https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt")
        _lama[device] = torch.jit.load(mp, map_location=device).eval().to(device)
    return _lama[device]


def _dashscope_key():
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key and (ROOT / ".env").exists():
        for ln in (ROOT / ".env").read_text().splitlines():
            if ln.startswith("DASHSCOPE_API_KEY="):
                key = ln.split("=", 1)[1].strip().strip('"').strip("'")
    return key


# ---------------- 文字层 ----------------
def _measure_color(rgb, x, y, w, h):
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = max(0, x - 2), max(0, y - 2), min(W, x + w + 2), min(H, y + h + 2)
    crop = rgb[y0:y1, x0:x1].astype(float)
    if crop.size == 0:
        return "#000000"
    b = np.concatenate([crop[0:2].reshape(-1, 3), crop[-2:].reshape(-1, 3),
                        crop[:, 0:2].reshape(-1, 3), crop[:, -2:].reshape(-1, 3)])
    dist = np.linalg.norm(crop - np.median(b, axis=0), axis=2)
    dmax = dist.max()
    if dmax < 18:
        return "#000000"
    core, ink = dist > dmax * 0.70, dist > dmax * 0.35
    pix = crop.astype(np.uint8)[core if core.sum() >= 5 else ink]
    return "#%02x%02x%02x" % tuple(np.median(pix, axis=0).astype(int))


def _vlm_classify(image_path, texts, W, H, model=None):
    """VLM 文字分类。model/endpoint 留空时读环境变量, 便于换模型/换中转。
    返回状态串: ok / skipped:no_key / skipped:no_text / failed:<reason>。"""
    model = model or os.getenv("DASHSCOPE_VLM_MODEL", "qwen-vl-max")
    endpoint = os.getenv("DASHSCOPE_VLM_ENDPOINT", ENDPOINT)
    key = _dashscope_key()
    if not key:
        return "skipped:no_key"
    if not texts:
        return "skipped:no_text"
    import httpx
    listing = "\n".join(f'{i}. "{t["text"]}"' for i, t in enumerate(texts))
    prompt = (f"图像 {W}x{H}。下面是OCR识别到的文字(可能有错字)。对每条结合图像判断 "
              f"font_weight(300/400/700)、font_family(中文如'思源黑体',西文'sans-serif'/'serif')、并给更正后的 corrected。"
              f"严格只输出JSON数组,每项 {{\"i\":序号,\"corrected\":\"\",\"font_weight\":整数,\"font_family\":\"\"}}。\n" + listing)
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    payload = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]}
    try:
        r = httpx.post(endpoint, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=180)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        mm = re.search(r"\[.*\]", content, re.S)
        for item in json.loads(re.sub(r",\s*([}\]])", r"\1", mm.group(0)) if mm else content):
            i = item.get("i")
            if isinstance(i, int) and 0 <= i < len(texts):
                if item.get("font_weight"):
                    texts[i]["font_weight"] = int(item["font_weight"])
                if item.get("font_family"):
                    texts[i]["font_family_guess"] = item["font_family"]
                if item.get("corrected"):
                    texts[i]["text_corrected"] = item["corrected"]
        return "ok"
    except Exception as e:
        print("VLM classify 跳过:", e)
        return "failed:" + str(e)[:80]


def text_layer(image_path, use_vlm=True, min_score=0.5):
    img = Image.open(image_path).convert("RGBA")
    W, H = img.size
    rgb = np.array(Image.alpha_composite(Image.new("RGBA", img.size, (255,) * 4), img).convert("RGB"))
    res, _ = get_ocr()(str(image_path))
    texts = []
    for box, txt, score in (res or []):
        score = float(score)            # 新版 rapidocr 的 score 是字符串
        if score < min_score:
            continue
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        x, y = int(min(xs)), int(min(ys)); w, h = int(max(xs) - x), int(max(ys) - y)
        texts.append({"text": txt, "bbox": [x, y, w, h], "ocr_score": round(float(score), 2),
                      "font_size_px": max(1, round(h * SIZE_K)), "color": _measure_color(rgb, x, y, w, h),
                      "font_weight": 400, "font_family_guess": "思源黑体"})
    vlm_status = "disabled"
    if use_vlm:
        vlm_status = _vlm_classify(image_path, texts, W, H)
    return texts, (W, H), vlm_status


# ---------------- 前景层 ----------------
def foreground_layers(image_path, workdir, stem, model="birefnet-general",
                      min_area=0.004, close_ksize=7, alpha_thr=30, pad=4):
    """前景元素切分。
      min_area    : 最小连通域面积占比, 越小保留越多小元素(也更易出噪点)。
      close_ksize : 形态学闭运算核大小, <=1 时关闭闭运算(避免把相邻元素粘成一团)。
      alpha_thr   : 前景 alpha 二值化阈值。
      pad         : 元素裁剪外扩像素。
    默认值保持原行为(0.004/7/30/4), 异步服务路径不受影响; 节点按需传更细的值。"""
    from rembg import remove
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    fg = np.array(remove(img, session=get_rembg(model)))   # RGBA
    alpha = fg[:, :, 3]
    m = (alpha > alpha_thr).astype(np.uint8) * 255
    if close_ksize and int(close_ksize) > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_ksize), int(close_ksize)))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    eldir = Path(workdir) / "elements"; eldir.mkdir(parents=True, exist_ok=True)
    images = []
    for lab in range(1, n):
        x, y, w, h, area = stats[lab]
        if area < min_area * W * H:
            continue
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        comp = fg.copy(); comp[:, :, 3] = np.where(labels == lab, fg[:, :, 3], 0)
        fn = f"elements/el_{len(images)}.png"
        Image.fromarray(comp[y0:y1, x0:x1], "RGBA").save(Path(workdir) / fn)
        images.append({"url": fn, "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]})
    return images, (alpha > alpha_thr)


# ---------------- 背景层 ----------------
def background_layer(image_path, fg_bool_mask, texts, workdir, stem, dilate=13, it=3):
    """全分辨率构建 mask; LaMa 填充对超大图(4K)内部降采样再放大回原尺寸,
    避免 OOM, 同时背景层仍与全分辨率前景/文字层对齐。"""
    import torch
    prepare_img_and_mask, _ = _lama_utils()
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    mask = (fg_bool_mask.astype(np.uint8) * 255)
    for t in texts:
        x, y, w, h = t["bbox"]
        mask[max(0, y):min(H, y + h), max(0, x):min(W, x + w)] = 255
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)), iterations=it)

    # 大图: LaMa 在 LAMA_MAX_SIDE 上跑, 结果再放大回 (W,H)。CPU默认2048, GPU可调大。
    lama_max = int(os.getenv("LAMA_MAX_SIDE", "2048"))
    inp_img, inp_mask = img, Image.fromarray(mask).convert("L")
    scaled = max(W, H) > lama_max
    if scaled:
        s = lama_max / max(W, H)
        sw, sh = int(W * s), int(H * s)
        inp_img = img.resize((sw, sh), Image.LANCZOS)
        inp_mask = inp_mask.resize((sw, sh), Image.NEAREST)

    def _run(dev):
        it_t, mt = prepare_img_and_mask(inp_img, inp_mask, torch.device(dev))
        with torch.inference_mode():
            return get_lama(dev)(it_t, mt)[0].permute(1, 2, 0).detach().cpu().numpy()
    dev = get_device()
    try:
        arr = _run(dev)
    except Exception as e:        # MPS 等不支持某算子 -> 回退CPU(模型/质量不变)
        if dev != "cpu":
            print(f"LaMa {dev} 推理失败, 回退 CPU(质量不变): {str(e)[:90]}")
            arr = _run("cpu")
        else:
            raise
    bg = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8)).convert("RGB")
    # LaMa 直接调模型(绕过 SimpleLama.__call__), 输出被 pad 到 8 的倍数; 先裁回喂入尺寸,
    # 再(若缩放过)放大回原尺寸, 保证 background 与 canvas/图层严格同尺寸。
    bg = bg.crop((0, 0, inp_img.width, inp_img.height))
    if scaled:
        bg = bg.resize((W, H), Image.LANCZOS)   # 放大回原尺寸, 与全分辨率图层对齐
    bgfn = "background.png"
    bg.save(Path(workdir) / bgfn)
    return bgfn


# ---------------- 主流程 ----------------
def run_pipeline(image_path, workdir, stem="img", use_vlm=True, fg_model="birefnet-general",
                 fg_kwargs=None, text_min_score=0.5):
    """4K 鲁棒: 全流程在 ≤WORK_MAX_SIDE 的工作分辨率上计算(防 OOM),
    最后把坐标/画布/背景放回原始(真4K)分辨率。前景元素层保持工作分辨率(放大显示)。
    fg_kwargs: 透传给 foreground_layers 的切分参数(min_area/close_ksize/alpha_thr/pad)。"""
    Path(workdir).mkdir(parents=True, exist_ok=True)
    orig = Image.open(image_path).convert("RGB")
    W0, H0 = orig.size
    work_max = int(os.getenv("WORK_MAX_SIDE", "2048"))
    scale, work_path = 1.0, image_path
    if max(W0, H0) > work_max:
        scale = work_max / max(W0, H0)
        wp = Path(workdir) / "work.png"
        orig.resize((round(W0 * scale), round(H0 * scale)), Image.LANCZOS).save(wp)
        work_path = str(wp)

    texts, _, vlm_status = text_layer(work_path, use_vlm=use_vlm, min_score=text_min_score)
    images, fg_mask = foreground_layers(work_path, workdir, stem, model=fg_model, **(fg_kwargs or {}))
    bg = background_layer(work_path, fg_mask, texts, workdir, stem)

    if scale != 1.0:                       # 坐标/产物放回原始 4K 空间
        inv = 1.0 / scale
        for t in texts:
            t["bbox"] = [round(v * inv) for v in t["bbox"]]
            t["font_size_px"] = max(1, round(t["font_size_px"] * inv))
        for im in images:
            im["bbox"] = [round(v * inv) for v in im["bbox"]]
        bgp = Path(workdir) / bg
        Image.open(bgp).resize((W0, H0), Image.LANCZOS).save(bgp)

    manifest = {"canvas": {"width": W0, "height": H0}, "background": bg, "images": images, "texts": texts,
                "meta": {"vlm_status": vlm_status}}
    json.dump(manifest, open(Path(workdir) / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return manifest


def run_pipeline_phased(image_path, workdir, stem, gpu_lock, use_vlm=True, fg_model="birefnet-general",
                        fg_kwargs=None, text_min_score=0.5):
    """异步服务用: 把流程拆成「GPU/模型段(加锁串行)」和「VLM网络段(不锁, 跨任务重叠)」。
    GPU 物理本就串行 + 模型单例非线程安全 -> GPU 段必须加锁; VLM(~34s网络)是 GPU 空闲期 ->
    不加锁让多任务在此重叠, 榨干单卡吞吐。其余 4K 缩放/坐标换算与 run_pipeline 完全一致。"""
    Path(workdir).mkdir(parents=True, exist_ok=True)
    orig = Image.open(image_path).convert("RGB")
    W0, H0 = orig.size
    work_max = int(os.getenv("WORK_MAX_SIDE", "2048"))
    scale, work_path = 1.0, image_path
    if max(W0, H0) > work_max:
        scale = work_max / max(W0, H0)
        wp = Path(workdir) / "work.png"
        orig.resize((round(W0 * scale), round(H0 * scale)), Image.LANCZOS).save(wp)
        work_path = str(wp)
    ww, wh = Image.open(work_path).size

    vlm_status = "disabled"
    with gpu_lock:                          # ① OCR+测色(模型) 串行
        texts, _, _ = text_layer(work_path, use_vlm=False, min_score=text_min_score)
    if use_vlm:                             # ② VLM(网络) 不锁 -> 多任务重叠
        vlm_status = _vlm_classify(work_path, texts, ww, wh)
    with gpu_lock:                          # ③ 前景BiRefNet + 背景LaMa(模型/GPU) 串行
        images, fg_mask = foreground_layers(work_path, workdir, stem, model=fg_model, **(fg_kwargs or {}))
        bg = background_layer(work_path, fg_mask, texts, workdir, stem)

    if scale != 1.0:                        # 坐标/产物放回原始尺寸
        inv = 1.0 / scale
        for t in texts:
            t["bbox"] = [round(v * inv) for v in t["bbox"]]
            t["font_size_px"] = max(1, round(t["font_size_px"] * inv))
        for im in images:
            im["bbox"] = [round(v * inv) for v in im["bbox"]]
        bgp = Path(workdir) / bg
        Image.open(bgp).resize((W0, H0), Image.LANCZOS).save(bgp)

    manifest = {"canvas": {"width": W0, "height": H0}, "background": bg, "images": images, "texts": texts,
                "meta": {"vlm_status": vlm_status}}
    json.dump(manifest, open(Path(workdir) / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return manifest


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--workdir", default="output/pipeline_test")
    ap.add_argument("--no-vlm", action="store_true")
    a = ap.parse_args()
    mf = run_pipeline(a.image, a.workdir, Path(a.image).stem, use_vlm=not a.no_vlm)
    print(json.dumps({"canvas": mf["canvas"], "background": mf["background"],
                      "n_images": len(mf["images"]), "n_texts": len(mf["texts"])}, ensure_ascii=False))
