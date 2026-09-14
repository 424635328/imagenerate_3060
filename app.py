"""app.py v3 — UX 完全体。运行: python app.py  (http://127.0.0.1:7860)
预制场景/风格 · Surprise · 1/4张网格 · 多风格对比 · 历史收藏画廊·重roll · 4x超分 ·
img2img(强度/去色/遮罩) · 复制seed · 负向编辑。
"""
import os, random, time
import numpy as np
import torch
from PIL import Image, ImageOps
from config import (
    ADAPTER_DIR, BASE_MODEL_DIR, HF_CACHE, LCM_DIR, SR_DIR, TCD_DIR,
    clamp_cfg, clamp_steps,
)

# —— 规范化：全程离线加载（权重均已本地化）——
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
# 进度条：默认开启（终端里能看加载/去噪进度）；设为 0 / false 可完全静音日志
_SHOW_PROGRESS = str(os.environ.get("PROGRESS_BAR", "1")).lower() not in ("0", "false", "off", "no")

# —— 基础模型：优先本地 safetensors，路径由 config.py 统一派生 ——
LOCAL_BASE = str(BASE_MODEL_DIR)
HF_BASE = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
BASE = os.environ.get("BASE_MODEL") or (LOCAL_BASE if BASE_MODEL_DIR.is_dir() else HF_BASE)
ADAPTER = str(ADAPTER_DIR)
LCM_DIR = str(LCM_DIR)
TCD_DIR = str(TCD_DIR)
# 超分模型统一由 enhance.SR_MODELS 注册表提供（见 get_sr()），这里不再硬编码路径；
# 需要回退商用权重时设 SR_MODEL=ultrasharp / realesrgan。

RANDOM_TEMPLATES = [
    "a scenic landscape, golden hour light, dramatic clouds",
    "a rugged mountain valley at sunrise, mist, alpine lake",
    "a tropical island coastline, turquoise sea, palm trees",
    "a lush green rainforest river, mossy rocks, sunlight rays",
    "a vast golden desert with dunes under a clear blue sky",
    "a majestic waterfall in a mossy gorge, flowing water",
    "a tranquil alpine lake reflecting snow-capped peaks",
    "a wildflower meadow under a towering peak, spring",
    "a coastal cliff at sunset, waves crashing, warm light",
    "an aerial view of a winding river through green valleys",
]
SCENE = {
    # 自然奇观
    "山川日出": "a dramatic mountain valley at sunrise, mist, alpine lake, golden light",
    "雪山湖泊": "a tranquil alpine lake reflecting snow-capped peaks, morning fog",
    "大峡谷": "a vast red rock canyon with layered cliffs, a river far below, dramatic light",
    "冰川峡湾": "a glacier-carved fjord with steep cliffs, calm dark water, low clouds",
    "火山地貌": "a volcanic landscape with black lava fields, steaming craters, moody sky",
    "巨石海岸": "a rugged coastline with tall sea stacks and crashing waves at sunset",
    "雨林瀑布": "a lush green rainforest waterfall, mossy rocks, sun rays through the canopy",
    "梯田云海": "terraced rice fields in mountain mist at dawn, layered green hills",
    # 海岸与水
    "热带海岸": "a serene turquoise tropical coastline, palm trees, calm sea, golden hour",
    "珊瑚礁海": "a crystal clear tropical lagoon with coral reef, aerial view, turquoise water",
    "礁石浪花": "ocean waves crashing over dark rocks, sea spray, long exposure, moody light",
    "白沙海岛": "a white sand island surrounded by turquoise water, aerial view, sunlight",
    "湖心倒影": "a calm lake at dusk with perfect mirror reflections of mountains and clouds",
    "瀑布深潭": "a powerful waterfall plunging into a teal pool in a lush mossy gorge",
    # 四季
    "秋色枫林": "an autumn forest with golden and red maple leaves, misty path, soft light",
    "春日花海": "a blooming wildflower meadow below snowy peaks, spring, clear blue sky",
    "冬雪森林": "a snow-covered pine forest under soft winter light, quiet and cold",
    "樱花湖畔": "cherry blossoms framing a calm lake with mountains behind, spring",
    "夏日草原": "rolling green grassland with scattered trees and big white clouds",
    "薰衣草田": "rows of lavender fields leading to a distant village at sunset",
    # 光效与天象
    "极光雪原": "aurora borealis over a snowy landscape, winter night, stars, green glow",
    "银河星空": "the milky way over a desert rock formation, starry night, long exposure",
    "云海日出": "a sea of clouds below mountain peaks at sunrise, warm golden light",
    "金色日落": "a coastline at golden sunset, warm light, glowing reflections on water",
    "晨雾森林": "a misty forest at dawn with light rays through the trees, dew",
    "暴风乌云": "dramatic storm clouds over open plains, moody light, distant rain",
    "彩虹山谷": "a rainbow over a green valley after rain, wet grass, soft light",
    "沙漠日落": "golden sand dunes with long shadows at sunset, rippled texture",
}
STYLES = {
    "写实摄影": "", "油画": "oil painting, thick impasto brushstrokes, vibrant van gogh colors",
    "水墨": "chinese ink wash painting, sumi-e, ink on rice paper, minimalist",
    "赛博朋克": "cyberpunk, neon purple and blue lights, rain, futuristic, cinematic",
    "水彩": "watercolor painting, soft washes, loose brushwork",
    "丙烯印象": "acrylic impressionist painting, textured strokes, dramatic sky",
    "动漫": "anime style, studio ghibli, vibrant, soft light",
}
DEFAULT_NEG = "blurry, low quality, watermark, text, distorted, oversaturated, bad anatomy"
_pipe=None; _i2i=None; _sr=None
_pipe_mode=None          # (fast, sampler)，同一时间只驻留一条管线
_i2i_mode=None

# SD1.5 潜空间 → RGB 的线性近似（ComfyUI 同款系数）。
# 只用于「生成中」的低成本缩略预览：不过 VAE，单步开销 < 2 ms。
_LATENT_RGB = [
    [0.3512, 0.2297, 0.3227],
    [0.3250, 0.4974, 0.2350],
    [-0.2829, 0.1762, 0.2721],
    [-0.2120, -0.2616, -0.7177],
]


def latent_preview(latents, max_edge: int = 320) -> Image.Image:
    """把去噪中的 latent 近似解码成一张小图（用于渐进预览）。"""
    with torch.no_grad():
        matrix = torch.tensor(_LATENT_RGB, dtype=torch.float32, device=latents.device)
        rgb = latents[0].float().permute(1, 2, 0) @ matrix
        rgb = ((rgb + 1.0) / 2.0).clamp(0, 1).mul(255).round().byte().cpu().numpy()
    img = Image.fromarray(rgb)
    scale = max(1, max_edge // max(img.size))
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.Resampling.BILINEAR)
    return img


def unload_pipes() -> None:
    """彻底释放管线（空闲回收 / 手动 GC 用），下次请求会重新懒加载。"""
    global _pipe, _i2i, _pipe_mode, _i2i_mode, _sr
    for obj in (_pipe, _i2i):
        if obj is None:
            continue
        try:
            obj.remove_all_hooks()
        except Exception:
            pass
    _pipe = None; _i2i = None; _pipe_mode = None; _i2i_mode = None; _sr = None
    from runtime import collect_gpu
    collect_gpu(deep=True)


def pipe_loaded() -> bool:
    return _pipe is not None


def current_mode() -> dict:
    return {"loaded": _pipe is not None,
            "fast": bool(_pipe_mode[0]) if _pipe_mode else None,
            "sampler": _pipe_mode[1] if _pipe_mode else None,
            # 当前驻留的是哪份权重（前端要显示"正在用哪个版本"）
            "adapter": _pipe_mode[2] if _pipe_mode and len(_pipe_mode) > 2 else None}


def warmup(fast: bool = False, sampler: str = "dpmpp2m_karras") -> dict:
    """预热管线：把首张图的 ~30 s 冷启动挪到用户点「生成」之前。"""
    t0 = time.time()
    get_pipe(fast, sampler)
    return {"loaded": True, "seconds": round(time.time() - t0, 2),
            "fast": bool(fast), "sampler": (sampler or "dpmpp2m_karras").lower()}


def _configure_memory(pipe):
    """启用低显存路径；优先 xFormers，失败则使用 PyTorch SDPA/切片。"""
    try:
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    except Exception:
        pass
    if os.environ.get("USE_XFORMERS", "0").lower() in {"1", "true", "yes"}:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            return
        except Exception as exc:
            print("xformers unavailable; using PyTorch attention:", exc)
    # PyTorch 2.x uses scaled_dot_product_attention (SDPA) in diffusers.
    try:
        pipe.enable_attention_slicing("auto")
    except Exception:
        pass


def _set_scheduler(pipe, sampler: str, fast: bool):
    from diffusers import (
        DDIMScheduler, DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LCMScheduler,
    )
    name = (sampler or "dpmpp2m_karras").lower()
    if fast or name == "lcm":
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        return "lcm"
    if name == "tcd":
        try:
            from diffusers import TCDScheduler
            pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
            return "tcd"
        except ImportError:
            print("TCDScheduler unavailable; falling back to LCM scheduler")
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            return "lcm"
    schedulers = {
        "dpmpp2m": (DPMSolverMultistepScheduler, {}),
        "dpmpp2m_karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
        "dpmpp2m_sde": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True}),
        "euler_a": (EulerAncestralDiscreteScheduler, {}),
        "euler": (EulerDiscreteScheduler, {}),
        "ddim": (DDIMScheduler, {}),
    }
    cls, kwargs = schedulers.get(name, schedulers["dpmpp2m_karras"])
    pipe.scheduler = cls.from_config(pipe.scheduler.config, **kwargs)
    return name if name in schedulers else "dpmpp2m_karras"


def _build_pipe(fast: bool, sampler: str = "dpmpp2m_karras", adapter_dir: str | None = None):
    """构建质量或少步管线；fast 先合并风格 LoRA，再叠加 LCM/TCD LoRA。

    `adapter_dir` 为 None 时用 config 里的默认权重（Gradio 入口就是这种用法）；
    Web API 会按请求传具体版本目录（slug 已在服务端白名单里解析过，这里不再拼接）。
    """
    from diffusers import StableDiffusionPipeline
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    if not _SHOW_PROGRESS:
        try:
            from diffusers.utils import logging as _dl; _dl.disable_progress_bar()
            from transformers.utils import logging as _tl; _tl.disable_progress_bar()
        except Exception: pass
    adapter_path = str(adapter_dir or ADAPTER)
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE, dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    adapter_cfg = LoraConfig.from_pretrained(adapter_path)
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if fast or sampler.lower() in {"lcm", "tcd"}:
        pipe.unet = get_peft_model(pipe.unet, adapter_cfg)
        set_peft_model_state_dict(pipe.unet, load_file(adapter_file))
        pipe.unet = pipe.unet.merge_and_unload()
        distill_dir = TCD_DIR if sampler.lower() == "tcd" and os.path.isdir(TCD_DIR) else LCM_DIR
        distill_file = "pytorch_lora_weights.safetensors"
        if os.path.isdir(distill_dir):
            pipe.load_lora_weights(distill_dir, weight_name=distill_file)
            print(f"[mode] {sampler.upper()} few-step, steps 4~{12 if fast else 8}")
        else:
            print("[mode] distillation LoRA missing:", distill_dir)
    else:
        pipe.unet = get_peft_model(pipe.unet, adapter_cfg)
        set_peft_model_state_dict(pipe.unet, load_file(adapter_file))
        print(f"[mode] QUALITY sampler: {sampler} adapter: {adapter_path}")
    actual_sampler = _set_scheduler(pipe, sampler, fast)
    te = str(adapter_path) + "_text_encoder.pt"
    if os.path.exists(te):
        pipe.text_encoder.load_state_dict(torch.load(te, map_location="cpu"))
        print("loaded fine-tuned text_encoder:", te)
    else:
        # 没有微调过的 TE 就必须用基座的 —— 否则上一个版本的 TE 会串味到这一次
        print("[mode] text_encoder: base (no fine-tuned file for this adapter)")
    _configure_memory(pipe)
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pipe = pipe.to("cuda")
    pipe._landscape_sampler = actual_sampler
    pipe._landscape_adapter = adapter_path
    return pipe


def get_pipe(fast: bool = False, sampler: str = "dpmpp2m_karras", adapter_dir: str | None = None):
    """只驻留一条 (模式, 采样器, 版本) 管线，切换时释放 hooks/引用/显存。

    ⚠️ 版本必须进缓存键：否则切了版本还命中旧管线 —— 本项目被"静默 no-op"咬过两次
    （`load_lora_weights` 不加载 peft 权重、PEFT 融合产出全零 lora_B），
    表现都是"键与形状都对、出图却和基座逐位相同"。
    """
    global _pipe, _i2i, _pipe_mode, _i2i_mode
    key = (bool(fast), (sampler or "dpmpp2m_karras").lower(), str(adapter_dir or ADAPTER))
    if _pipe is None or _pipe_mode != key:
        if _pipe is not None:
            try: _pipe.remove_all_hooks()
            except Exception: pass
            _pipe = None; _i2i = None; _i2i_mode = None
            from runtime import collect_gpu
            collect_gpu(deep=True)
        _pipe = _build_pipe(*key); _pipe_mode = key
    return _pipe


def get_i2i(fast: bool = False, sampler: str = "dpmpp2m_karras", adapter_dir: str | None = None):
    global _i2i, _i2i_mode
    key = (bool(fast), (sampler or "dpmpp2m_karras").lower(), str(adapter_dir or ADAPTER))
    if _i2i is None or _i2i_mode != key:
        from diffusers import StableDiffusionImg2ImgPipeline
        p = get_pipe(*key)
        _i2i = StableDiffusionImg2ImgPipeline(
            vae=p.vae, text_encoder=p.text_encoder, tokenizer=p.tokenizer,
            unet=p.unet, scheduler=p.scheduler,
            feature_extractor=getattr(p, "feature_extractor", None),
            safety_checker=None, requires_safety_checker=False)
        try: _i2i.enable_model_cpu_offload()
        except Exception: _i2i = _i2i.to("cuda")
        _i2i_mode = key
    return _i2i

def get_sr():
    """加载超分模型 —— 走 `enhance.SR_MODELS` 同一注册表，三个入口（Gradio / API / 前端）保持一致。

    默认用**本项目自训**的 `ours`（风景领域微调，细节量 6.53×/8.69× vs bicubic，
    见 README §6.3 与 docs/TRAINING.md §8）；可用 `SR_MODEL` 环境变量切回商用权重。
    原来这里硬编码 RealESRGAN，导致网页端与 API 端默认不是同一个模型。
    """
    global _sr
    if _sr is None:
        import enhance
        _sr = enhance.get_sr(os.environ.get("SR_MODEL", "ours"))
    return _sr

def _sr_up(img):
    m = get_sr().eval().to("cuda")
    x = np.asarray(img.convert("RGB")).astype(np.float32)/255.0
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).to("cuda")
    with torch.no_grad():
        o = m(x).clamp(0,1).squeeze(0).permute(1,2,0).cpu().numpy()
    result = Image.fromarray((o*255).astype(np.uint8))
    # Keep the cached model on CPU between jobs so SR does not compete with UNet.
    m.to("cpu")
    del x, o
    from runtime import collect_gpu
    collect_gpu()
    return result

def _size(res, aspect):
    res=int(res)
    if aspect=="横": return int(res*1.5), res
    if aspect=="竖": return res, int(res*1.5)
    return res, res

def _grid(imgs, cols=None):
    cols = cols or (2 if len(imgs)>=2 else 1)
    w,h = imgs[0].size; rows=(len(imgs)+cols-1)//cols
    c=Image.new("RGB",(w*cols,h*rows),(10,10,10))
    for i,im in enumerate(imgs):
        r,co=divmod(i,cols); c.paste(im,(co*w,r*h))
    return c

def _composite(ref, gen, mask):
    m = np.asarray(mask.convert("L").resize(gen.size)).astype(np.float32)/255.0
    m = np.stack([m]*3, axis=-1)
    r = np.asarray(ref.convert("RGB").resize(gen.size)).astype(np.float32)
    g_ = np.asarray(gen.convert("RGB")).astype(np.float32)
    return Image.fromarray(((r*(1-m)+g_*m)).astype(np.uint8))

def build_prompt(prompt, style, randomize):
    if randomize:
        prompt = random.choice(RANDOM_TEMPLATES)
    prompt = prompt.strip() or random.choice(RANDOM_TEMPLATES)
    if style and STYLES.get(style,""):
        prompt = f"{prompt}, {STYLES[style]}"
    return prompt

def _progress_hook(total_steps, progress_cb, preview_every, preview_cb):
    """构造 diffusers callback：上报步数进度，并按间隔产出 latent 近似预览。"""
    if progress_cb is None and preview_cb is None:
        return None

    def _cb(pipe, step_index, timestep, kwargs):
        done = int(step_index) + 1
        if progress_cb is not None:
            try: progress_cb(done, total_steps)
            except Exception: pass
        if preview_cb is not None and preview_every > 0 and done < total_steps and done % preview_every == 0:
            latents = kwargs.get("latents")
            if latents is not None:
                try: preview_cb(latent_preview(latents), done, total_steps)
                except Exception: pass
        return kwargs

    return _cb


def _gen_one(prompt, w, h, steps, cfg, seed, neg, refimg, strength, desat, mask,
             fast=False, sampler="dpmpp2m_karras", progress_cb=None,
             preview_cb=None, preview_every=0, adapter_dir=None):
    fast = bool(fast)
    steps = clamp_steps(steps, fast=fast)
    cfg = clamp_cfg(cfg, fast=fast)
    pipe = get_pipe(fast, sampler, adapter_dir)
    g = torch.Generator(device="cuda").manual_seed(seed)
    kw = dict(prompt=prompt, negative_prompt=neg, num_inference_steps=steps,
              guidance_scale=cfg, width=w, height=h, generator=g)
    hook = _progress_hook(steps, progress_cb, preview_every, preview_cb)
    if hook is not None:
        kw["callback_on_step_end"] = hook
        kw["callback_on_step_end_tensor_inputs"] = ["latents"]
    if refimg is not None:
        ref = refimg.resize((w,h))
        if desat < 1.0:
            gray = ImageOps.grayscale(ref).convert("RGB")
            ref = Image.blend(ref, gray, 1.0-desat)
        # img2img 只跑 strength 比例的步数，进度总数据要相应缩短。
        i2i_kw = dict(kw)
        i2i_kw.pop("width", None); i2i_kw.pop("height", None)
        if hook is not None:
            i2i_kw["callback_on_step_end"] = _progress_hook(
                max(1, int(steps * min(1.0, strength))), progress_cb, preview_every, preview_cb)
        img = get_i2i(fast, sampler)(image=ref, strength=strength, **i2i_kw).images[0]
        if mask is not None:
            img = _composite(ref, img, mask)
    else:
        img = pipe(**kw).images[0]
    return img

def run(prompt, style, randomize, res, aspect, steps, cfg, seed, neg, batch, upscale,
        refimg, strength, desat, mask, hist, multi):
    hist = list(hist)
    try:
        res=int(res); aspect=str(aspect); seed=int(seed)
        steps=max(10,min(int(steps),80)); cfg=max(1.0,min(float(cfg),15.0))
        batch=max(1,min(int(batch),4)); strength=max(0.1,min(float(strength),0.95)); desat=max(0.0,min(float(desat),1.0))
        prompt = build_prompt(prompt, style, randomize)
        w,h = _size(res,aspect); t0=time.time()
        if multi:
            styles=["油画","水墨","赛博朋克","水彩"]; imgs=[]
            for i,st in enumerate(styles):
                p = f"{prompt}, {STYLES[st]}" if STYLES.get(st) else prompt
                imgs.append(_gen_one(p, w,h, steps, cfg, seed+i, neg, refimg, strength, desat, mask))
            if upscale: imgs=[_sr_up(x) for x in imgs]
            out=_grid(imgs,2)
            note=f"[多风格] {prompt}\n{w}×{h} · 4风格 · seed={seed}..{seed+3} · {time.time()-t0:.1f}s"
        else:
            imgs=[]
            for i in range(batch):
                imgs.append(_gen_one(prompt, w,h, steps, cfg, seed+i, neg, refimg, strength, desat, mask))
            if upscale: imgs=[_sr_up(x) for x in imgs]
            out=imgs[0] if len(imgs)==1 else _grid(imgs,2)
            note=f"seed={seed} · {w}×{h} · {len(imgs)}张 · {time.time()-t0:.1f}s · {'超分' if upscale else ''}\nprompt: {prompt}"
        hist.append((out,note))
        return out, note, hist, hist
    except Exception as e:
        return None, f"错误: {e}", hist, hist

def gen_wrap(prompt, style, randomize, res, aspect, steps, cfg, seed, neg, batch, upscale, refimg, strength, desat, mask, hist):
    return run(prompt, style, randomize, res, aspect, steps, cfg, seed, neg, batch, upscale, refimg, strength, desat, mask, hist, False)
def reroll_wrap(prompt, style, randomize, res, aspect, steps, cfg, neg, batch, upscale, refimg, strength, desat, mask, hist):
    return run(prompt, style, randomize, res, aspect, steps, cfg, -1, neg, batch, upscale, refimg, strength, desat, mask, hist, False)
def multi_wrap(prompt, style, randomize, res, aspect, steps, cfg, seed, neg, batch, upscale, refimg, strength, desat, mask, hist):
    return run(prompt, style, randomize, res, aspect, steps, cfg, seed, neg, batch, upscale, refimg, strength, desat, mask, hist, True)

def main():
    import gradio as gr
    with gr.Blocks(title="LANDSCAPE·ART",
                   theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple")) as demo:
        gr.Markdown("# 🏔 LANDSCAPE·ART 风景生图\n预制场景/风格 · **Surprise随机** · **1~4张网格** · **多风格对比** · **历史收藏画廊·重roll** · **4x超分** · **img2img(强度/去色/遮罩)** · **复制seed**。")
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="风格 prompt（留空=随机）", placeholder="e.g. a misty alpine lake at dawn", lines=2)
                style = gr.Dropdown(list(STYLES.keys()), value="写实摄影", label="风格")
                gr.Markdown("**预制场景（28 个）**")
                scene_btn = []
                _scenes = list(SCENE.items())
                for _i in range(0, len(_scenes), 7):
                    with gr.Row():
                        for _n, _p in _scenes[_i:_i+7]:
                            scene_btn.append(gr.Button(_n, size="sm"))
                randomize = gr.Checkbox(label="随机(Surprise)", value=False)
                with gr.Row():
                    res = gr.Radio(["512","768"], value="512", label="分辨率")
                    aspect = gr.Radio(["方","横","竖"], value="方", label="纵横比")
                    batch = gr.Radio(["1","4"], value="1", label="一次出图")
                with gr.Row():
                    steps = gr.Slider(20,60,value=40,step=5,label="步数")
                    cfg = gr.Slider(3.0,12.0,value=7.5,step=0.5,label="CFG")
                    seed = gr.Number(value=-1,label="seed(-1 随机)",precision=0)
                upscale = gr.Checkbox(label="4x 超分", value=False)
                neg = gr.Textbox(label="负向 prompt", value=DEFAULT_NEG, lines=2)
                with gr.Accordion("img2img（上传参考图）", open=False):
                    refimg = gr.Image(label="参考图", type="pil")
                    strength = gr.Slider(0.1,0.95,value=0.6,step=0.05,label="强度")
                    desat = gr.Slider(0.0,1.0,value=1.0,step=0.05,label="参考去色权重(1=保留颜色)")
                    mask = gr.Image(label="遮罩(白=重绘, 可选)", type="pil")
                with gr.Row():
                    genbtn = gr.Button("✨ 生成", variant="primary")
                    reroll = gr.Button("🎲 重roll(新seed)")
                    multibtn = gr.Button("🖼 多风格对比")
                    copybtn = gr.Button("📋 复制信息")
            with gr.Column(scale=2):
                out_img = gr.Image(label="输出", type="pil", interactive=False)
                out_txt = gr.Textbox(label="信息（seed/尺寸/耗时/prompt）", interactive=False)
                gallery = gr.Gallery(label="历史收藏画廊", columns=4, object_fit="contain")
        hist_state = gr.State([])
        def _set(scene): return lambda: SCENE[scene]
        for n, b in zip(SCENE.keys(), scene_btn):
            b.click(_set(n), inputs=[], outputs=[prompt])
        base_in = [prompt, style, randomize, res, aspect, steps, cfg, seed, neg, batch, upscale, refimg, strength, desat, mask, hist_state]
        genbtn.click(gen_wrap, inputs=base_in, outputs=[out_img, out_txt, hist_state, gallery])
        reroll_in = [prompt, style, randomize, res, aspect, steps, cfg, neg, batch, upscale, refimg, strength, desat, mask, hist_state]
        reroll.click(reroll_wrap, inputs=reroll_in, outputs=[out_img, out_txt, hist_state, gallery])
        multibtn.click(multi_wrap, inputs=base_in, outputs=[out_img, out_txt, hist_state, gallery])
        copybtn.click(lambda x: x, inputs=[out_txt], outputs=[], js="(x)=>{try{navigator.clipboard.writeText(x||'');}catch(e){}}")
        gr.Markdown("> 生成后显示 **seed/prompt/尺寸/耗时**；历史画廊可回看/下载；点 **重roll** 用当前 prompt 换新 seed；**多风格对比** 同一场景出 4 种风格。")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=False)
    print("Gradio app running at http://127.0.0.1:7860")

if __name__ == "__main__":
    main()
