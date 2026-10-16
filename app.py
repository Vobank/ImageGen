# -*- coding: utf-8 -*-
r"""
Генератор картинок на Stable Diffusion XL — локально, на твоей RTX 5080.
Пишешь описание → получаешь картинку. Плюс режим «переделать фото» (img2img).
Запуск: venv\Scripts\python.exe app.py  → http://localhost:5001
"""
import os
import io
import time
import base64
import threading

# анти-фрагментация видеопамяти (важно на 16 ГБ, когда открыты игры/Zoom/браузер)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print("Запускаю генератор картинок... грузится движок, подожди ~15-30 сек.", flush=True)

import torch
import requests
from flask import Flask, request, jsonify, send_file

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "images")
os.makedirs(OUT, exist_ok=True)

def _ensure_ollama():
    """Поднять Ollama, если не запущена (нужна для умного улучшения промпта)."""
    try:
        requests.get("http://localhost:11434/api/tags", timeout=3)
        return
    except Exception:
        pass
    import subprocess
    for p in [os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama app.exe")]:
        if os.path.exists(p):
            try:
                env = dict(os.environ); env.setdefault("OLLAMA_MODELS", r"D:\Ollama\models")
                subprocess.Popen([p], env=env, creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
                for _ in range(20):
                    time.sleep(1)
                    try:
                        requests.get("http://localhost:11434/api/tags", timeout=2); return
                    except Exception:
                        pass
            except Exception:
                pass

# Доступные модели: ключ -> (название, источник)
# источник = строка (репозиторий diffusers) ИЛИ кортеж (репо, имя одиночного .safetensors)
MODELS = {
    "real5": ("📸 Фотореализм V5 (макс.)", ("SG161222/RealVisXL_V5.0", "RealVisXL_V5.0_fp16.safetensors")),
    "real":  ("📸 Фотореализм V4",          "SG161222/RealVisXL_V4.0"),
    "art":   ("🎨 Художественный",           "stabilityai/stable-diffusion-xl-base-1.0"),
}
DEFAULT_MODEL = "real5"  # RealVisXL V5 — проверено, лучший фотореализм

# Базовый «стоп-лист»: то, что приложение ВСЕГДА исключает само (плохое качество,
# кривая анатомия/лица, пластик, артефакты, кривая геометрия фона). Пользователь
# может дописать своё сверху — оно добавляется к этому, а не заменяет.
BASE_NEGATIVE = ("lowres, blurry, worst quality, jpeg artifacts, bad anatomy, "
                 "deformed, ugly, extra limbs, extra fingers, mutated hands, plastic skin, "
                 "airbrushed, overprocessed, lifeless, dull, oversaturated, cartoonish, "
                 "3d render, cgi, illustration, painting, doll, watermark, text, "
                 "deformed face, bad eyes, asymmetric eyes, disfigured face, "
                 "warped geometry, distorted perspective, crooked lines, broken architecture, "
                 "deformed railings, melting structures, incoherent background, "
                 "misaligned edges, wavy straight lines, duplicated objects")

def _single_file_path(repo, filename):
    """Путь к уже скачанному одиночному .safetensors (без докачки, только из кэша)."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=filename, local_files_only=True)

app = Flask(__name__)

_txt = None
_active = None            # какая модель сейчас загружена
_lock = threading.Lock()
_status = {"loading": False, "loaded": False, "error": ""}
_progress = {"stage": "", "step": 0, "total": 0}   # живой прогресс генерации

def _set_stage(stage, total=0):
    _progress.update({"stage": stage, "step": 0, "total": total})

def _step_cb(pipe, step, t, kw):
    _progress["step"] = step + 1
    return kw

_vae = None
_vae_tried = False
def _get_vae():
    """Улучшенный VAE (цвета/детали) — если скачан. Иначе None (берём встроенный)."""
    global _vae, _vae_tried
    if _vae is None and not _vae_tried:
        _vae_tried = True
        try:
            from diffusers import AutoencoderKL
            _vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,
                local_files_only=True).to("cuda")
        except Exception:
            _vae = None
    return _vae

def _ensure(model_key):
    """Загрузить нужную модель (в 16 ГБ влезает одна — при смене выгружаем старую)."""
    global _txt, _active, _status, _inpaint
    if model_key not in MODELS:
        model_key = DEFAULT_MODEL
    if _active == model_key and _txt is not None:
        return _txt
    # выгружаем предыдущую модель И инпейнт-модель, чистим видеопамять
    _txt = None
    _inpaint = None
    import gc
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    src = MODELS[model_key][1]
    _status["loading"] = True
    try:
        if isinstance(src, tuple):
            # одиночный файл .safetensors (напр. RealVisXL V5) -> from_single_file
            path = _single_file_path(*src)
            p = StableDiffusionXLPipeline.from_single_file(
                path, torch_dtype=torch.float16, use_safetensors=True,
                config="stabilityai/stable-diffusion-xl-base-1.0", local_files_only=True)
        else:
            try:
                p = StableDiffusionXLPipeline.from_pretrained(
                    src, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
            except Exception:
                p = StableDiffusionXLPipeline.from_pretrained(
                    src, torch_dtype=torch.float16, use_safetensors=True)
        p = p.to("cuda")
        p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config, use_karras_sigmas=True)
        v = _get_vae()                     # улучшенный VAE, если скачан
        if v is not None:
            p.vae = v
        try:
            p.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)   # FreeU — резче/детальнее, бесплатно
        except Exception:
            pass
        try:
            p.vae.enable_tiling(); p.vae.enable_slicing()    # меньше пик VRAM на декоде — не упасть по памяти
        except Exception:
            pass
        _txt = p
        _active = model_key
        _status["loaded"] = True
        _status["error"] = ""
    except Exception as e:
        _status["error"] = str(e)
        raise
    finally:
        _status["loading"] = False
    return _txt

def _get_img():
    from diffusers import StableDiffusionXLImg2ImgPipeline
    return StableDiffusionXLImg2ImgPipeline(**_txt.components).to("cuda")

def _get_inpaint():
    from diffusers import StableDiffusionXLInpaintPipeline
    return StableDiffusionXLInpaintPipeline(**_txt.components).to("cuda")

INPAINT_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"  # спец-модель для «Убрать»
_inpaint = None
def _ensure_inpaint():
    """Спец-модель инпейнта (чисто убирает объект и достраивает фон). В 16 ГБ влезает
    одна модель — выгружаем обычную. Если модель ещё не скачана — вернём None (фолбэк)."""
    global _inpaint, _txt, _active
    if _inpaint is not None:
        return _inpaint
    from diffusers import StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler
    # освобождаем видеопамять от обычной модели
    _txt = None; _active = None
    import gc; gc.collect()
    try: torch.cuda.empty_cache()
    except Exception: pass
    try:
        p = StableDiffusionXLInpaintPipeline.from_pretrained(
            INPAINT_ID, torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True, local_files_only=True).to("cuda")
        p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config, use_karras_sigmas=True)
        try: p.vae.enable_tiling(); p.vae.enable_slicing()
        except Exception: pass
        _inpaint = p
        return _inpaint
    except Exception:
        return None   # не скачана — вызывающий откатится на обычный способ

_lama = None
def _get_lama():
    """LaMa — чистое удаление объектов (достраивает фон, не выдумывает объекты)."""
    global _lama
    if _lama is None:
        from simple_lama_inpainting import SimpleLama
        _lama = SimpleLama()
    return _lama

def _lama_remove(init, mask):
    from PIL import Image
    out = _get_lama()(init.convert("RGB"), mask.convert("L"))
    out = out.convert("RGB")
    if out.size != init.size:
        out = out.resize(init.size, Image.LANCZOS)
    return out

_compel = None
_compel_on = None
def _get_compel():
    """Compel — поддержка ДЛИННЫХ промптов (без лимита 77 токенов). None если не установлен."""
    global _compel, _compel_on
    if _compel is None or _compel_on != _active:
        try:
            from compel import Compel, ReturnedEmbeddingsType
            _compel = Compel(
                tokenizer=[_txt.tokenizer, _txt.tokenizer_2],
                text_encoder=[_txt.text_encoder, _txt.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True], truncate_long_prompts=False)
            _compel_on = _active
        except Exception:
            _compel = None
    return _compel

def _gen(pipe, prompt, neg, **kw):
    """Генерация с поддержкой длинных промптов (compel), иначе обычно."""
    c = _get_compel()
    if c is not None:
        try:
            pe, pp = c(prompt)
            ne, npp = c(neg)
            pe, ne = c.pad_conditioning_tensors_to_same_length([pe, ne])
            return pipe(prompt_embeds=pe, pooled_prompt_embeds=pp,
                        negative_prompt_embeds=ne, negative_pooled_prompt_embeds=npp, **kw).images[0]
        except Exception:
            pass
    return pipe(prompt=prompt, negative_prompt=neg, **kw).images[0]

_ip_encoder = None
def _load_ipadapter():
    """Подключить IP-Adapter (генерация по фото-образцу) к текущей модели."""
    global _ip_encoder
    if _ip_encoder is None:
        from transformers import CLIPVisionModelWithProjection
        _ip_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", subfolder="sdxl_models/image_encoder",
            torch_dtype=torch.float16).to("cuda")
    _txt.image_encoder = _ip_encoder
    _txt.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                         weight_name="ip-adapter_sdxl.bin")

def _enhance_prompt(text, model_key="base"):
    """Локальный ИИ (qwen) превращает короткое описание (рус/англ) в богатый
    английский промпт с деталями, светом, ЭМОЦИЕЙ и качеством. keep_alive=0 —
    модель выгружается сразу, чтобы не занимать видеопамять при генерации."""
    text = (text or "").strip()
    if not text:
        return text
    style = ("RAW candid photograph, real life photo, shot on DSLR 50mm, natural imperfect lighting, "
             "realistic detailed skin with pores and texture, realistic hands and anatomy, real environment "
             "(streets, roads, natural background), subtle film grain, sharp focus, looks like an actual "
             "photo taken in real life, NOT cgi, NOT 3d render"
             if model_key in ("real", "real5")
             else "highly detailed, cinematic, dramatic atmospheric lighting, vivid, emotional, trending on artstation")
    sysmsg = ("You turn a short idea (in Russian or English) into ONE rich English prompt for a "
              "Stable Diffusion image. Translate meaning to English, add vivid concrete details, "
              "lighting, atmosphere, strong MOOD and EMOTION, composition, and quality tags "
              f"({style}, masterpiece, best quality, sharp focus, 8k). "
              "IMPORTANT: keep EVERY specific detail the user mentioned exactly — hair color, eye color, "
              "clothing, colors, objects, pose, actions — do NOT drop or change them. "
              "Output ONLY the final prompt on one line, no quotes, no explanations.")
    try:
        _ensure_ollama()
        r = requests.post("http://localhost:11434/api/chat", json={
            "model": "qwen2.5:7b-instruct",
            "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": text}],
            "stream": False, "keep_alive": 0, "options": {"temperature": 0.8},
        }, timeout=45)
        out = (r.json().get("message", {}).get("content") or "").strip().strip('"').replace("\n", " ")
        return out if len(out) > 5 else text
    except Exception:
        return text   # если Ollama недоступна — берём как есть

def _has_cyr(s):
    return any('а' <= c.lower() <= 'я' or c in 'ёЁ' for c in (s or ""))

def _translate_neg(text):
    """Перевести пользовательское «что ещё убрать» (рус) в короткие английские теги."""
    text = (text or "").strip()
    if not text:
        return ""
    try:
        _ensure_ollama()
        r = requests.post("http://localhost:11434/api/chat", json={
            "model": "qwen2.5:7b-instruct",
            "messages": [
                {"role": "system", "content": "Translate the user's list of things to AVOID/EXCLUDE from an "
                 "image into short English Stable Diffusion negative tags, comma-separated. "
                 "Output ONLY the tags on one line, no explanations."},
                {"role": "user", "content": text}],
            "stream": False, "keep_alive": 0, "options": {"temperature": 0.2},
        }, timeout=30)
        out = (r.json().get("message", {}).get("content") or "").strip().strip('"').replace("\n", " ")
        return out if len(out) > 1 else text
    except Exception:
        return text

def _build_negative(extra):
    """Итоговый негатив = базовый стоп-лист + (по желанию) пользовательские исключения."""
    extra = (extra or "").strip()
    if not extra:
        return BASE_NEGATIVE
    if _has_cyr(extra):
        extra = _translate_neg(extra)
    return BASE_NEGATIVE + ", " + extra

def _refine_faces(pil_img):
    """Найти лица и перерисовать каждое отдельно в высоком разрешении — реалистичнее."""
    try:
        import cv2, numpy as np
        arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces) == 0:
            return pil_img
        pipe = _get_img()
        W, H = pil_img.size
        for (x, y, w, h) in faces[:3]:
            pad = int(0.4 * w)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            crop = pil_img.crop((x0, y0, x1, y1))
            cw, ch = crop.size
            big = crop.resize((1024, 1024))
            ref = pipe(prompt="detailed realistic human face, sharp symmetric eyes, natural skin texture, photorealistic",
                       negative_prompt="deformed face, ugly, bad eyes, asymmetric eyes, blurry, disfigured, extra face",
                       image=big, strength=0.35, num_inference_steps=22, guidance_scale=5.5).images[0]
            pil_img.paste(ref.resize((cw, ch)), (x0, y0))
        return pil_img
    except Exception:
        return pil_img

def _photo_post(img, grain=0.035):
    """Лёгкое фото-зерно и микро-контраст — чтобы выглядело как снято камерой, а не 'пластик ИИ'."""
    try:
        import numpy as np
        from PIL import Image
        arr = np.asarray(img).astype(np.float32)
        noise = np.random.normal(0, 255 * grain, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    except Exception:
        return img

def _hires(img, prompt, neg, scale=1.5, strength=0.3):
    """Hires-fix: увеличить и дорисовать детали — вся картинка становится резче/реалистичнее."""
    try:
        from PIL import Image
        w, h = img.size
        big = img.resize((int(w * scale) // 8 * 8, int(h * scale) // 8 * 8), Image.LANCZOS)
        pipe = _get_img()
        c = _get_compel()
        if c is not None:
            try:
                pe, pp = c(prompt); ne, npp = c(neg)
                pe, ne = c.pad_conditioning_tensors_to_same_length([pe, ne])
                return pipe(prompt_embeds=pe, pooled_prompt_embeds=pp,
                            negative_prompt_embeds=ne, negative_pooled_prompt_embeds=npp,
                            image=big, strength=strength, num_inference_steps=20,
                            guidance_scale=5.0, callback_on_step_end=_step_cb).images[0]
            except Exception:
                pass
        return pipe(prompt=prompt, negative_prompt=neg, image=big, strength=strength,
                    num_inference_steps=20, guidance_scale=5.0,
                    callback_on_step_end=_step_cb).images[0]
    except Exception:
        return img

def _snap(x, default=1024):
    """Размер кадра: в разумных пределах и кратно 8 (требование SDXL)."""
    try:
        x = int(x)
    except Exception:
        x = default
    x = max(512, min(x, 1536))
    return x // 8 * 8

def _save_and_encode(img):
    name = f"img_{int(time.time()*1000)}.png"
    img.save(os.path.join(OUT, name))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), name

@app.route("/")
def index():
    return send_file(os.path.join(BASE, "index.html"))

@app.route("/status")
def status():
    return jsonify(_status)

@app.route("/progress")
def progress():
    return jsonify(_progress)

@app.route("/models")
def models():
    return jsonify({"models": [{"key": k, "name": v[0]} for k, v in MODELS.items()], "active": _active})

@app.route("/gallery")
def gallery():
    """Список сохранённых картинок (новые первыми)."""
    try:
        files = [f for f in os.listdir(OUT) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        files.sort(key=lambda f: os.path.getmtime(os.path.join(OUT, f)), reverse=True)
        return jsonify({"files": files[:120]})
    except Exception as e:
        return jsonify({"files": [], "error": str(e)})

@app.route("/images/<path:name>")
def image_file(name):
    """Отдать сохранённую картинку по имени."""
    name = os.path.basename(name)
    p = os.path.join(OUT, name)
    if os.path.exists(p):
        return send_file(p)
    return ("not found", 404)

@app.route("/generate", methods=["POST"])
def generate():
    d = request.json or {}
    prompt = (d.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Напиши, что нарисовать."})
    model = d.get("model", DEFAULT_MODEL)
    if d.get("enhance", True):                       # умное улучшение промпта
        _set_stage("Улучшаю запрос…")
        prompt = _enhance_prompt(prompt, model)
    _extra_neg = (d.get("negative") or "").strip()
    if _extra_neg and _has_cyr(_extra_neg):
        _set_stage("Готовлю исключения…")
    neg = _build_negative(_extra_neg)          # базовый стоп-лист + твоё (можно по-русски)
    steps = max(10, min(int(d.get("steps", 30)), 50))
    if model in ("real", "real5"):
        steps = max(steps, 34)                       # для реализма чуть больше шагов
    # формат кадра: либо width/height, либо квадрат size
    W = _snap(d.get("width") or d.get("size", 1024))
    H = _snap(d.get("height") or d.get("size", 1024))
    count = max(1, min(int(d.get("count", 1)), 4))    # сколько вариантов за раз
    seed_in = int(d.get("seed", -1))
    do_hires = bool(d.get("hires", False)) and count == 1   # hires только для 1 картинки (иначе долго)
    do_faces = bool(d.get("faces", True))
    results = []
    try:
        with _lock:
            _set_stage("Готовлю модель…")
            pipe = _ensure(model)
            gs = 5.0 if model in ("real", "real5") else 6.5   # ниже guidance = естественнее, менее "пластик"
            for i in range(count):
                seed = (seed_in + i) if seed_in >= 0 else (torch.seed() % (2**31))
                g = torch.Generator("cuda").manual_seed(seed)
                _set_stage(f"Рисую вариант {i+1}/{count}" if count > 1 else "Рисую", steps)
                img = _gen(pipe, prompt, neg, num_inference_steps=steps,
                           width=W, height=H, guidance_scale=gs, generator=g,
                           callback_on_step_end=_step_cb)
                if do_hires:
                    _set_stage("Увеличиваю детали", 20)
                    img = _hires(img, prompt, neg)
                if do_faces:
                    _set_stage("Улучшаю лица…")
                    img = _refine_faces(img)
                if model in ("real", "real5"):        # фото-зерно для эффекта реальной камеры
                    img = _photo_post(img)
                data, name = _save_and_encode(img)
                results.append({"image": data, "file": name, "seed": seed})
        _set_stage("", 0)
        # image/file — для совместимости (первый вариант), images — весь пакет
        return jsonify({"image": results[0]["image"], "file": results[0]["file"],
                        "seed": results[0]["seed"], "images": results})
    except Exception as e:
        _set_stage("", 0)
        return jsonify({"error": f"Ошибка генерации: {e}"})

@app.route("/img2img", methods=["POST"])
def img2img():
    d = request.json or {}
    prompt = (d.get("prompt") or "").strip()
    img_b64 = d.get("image") or ""
    if not prompt or not img_b64:
        return jsonify({"error": "Нужно фото и описание, что с ним сделать."})
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]
    if d.get("enhance", True):
        prompt = _enhance_prompt(prompt, d.get("model", DEFAULT_MODEL))
    strength = float(d.get("strength", 0.55))
    steps = max(10, min(int(d.get("steps", 30)), 50))
    try:
        from PIL import Image
        init = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        init = init.resize((1024, 1024))
        with _lock:
            _ensure(d.get("model", DEFAULT_MODEL))
            pipe = _get_img()
            img = pipe(prompt=prompt, negative_prompt=BASE_NEGATIVE, image=init, strength=strength,
                       num_inference_steps=steps, guidance_scale=6.5).images[0]
        data, name = _save_and_encode(img)
        return jsonify({"image": data, "file": name})
    except Exception as e:
        return jsonify({"error": f"Ошибка: {e}"})

@app.route("/inpaint", methods=["POST"])
def inpaint():
    """Убрать/заменить объект: image — фото, mask — где менять (белое), prompt — что там должно быть.
    Пусто = чистое удаление (достроить фон). С описанием = дорисовать новое."""
    d = request.json or {}
    img_b64 = d.get("image") or ""
    mask_b64 = d.get("mask") or ""
    user_prompt = (d.get("prompt") or "").strip()
    if not img_b64 or not mask_b64:
        return jsonify({"error": "Нужно фото и отметить область (закрасить объект)."})
    try:
        from PIL import Image, ImageFilter
        def _dec(b):
            if "," in b:
                b = b.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(b)))
        init = _dec(img_b64).convert("RGB")
        mask = _dec(mask_b64).convert("L").resize(init.size, Image.NEAREST)
        W0, H0 = init.size

        # === РЕЖИМ 1: пусто = ЧИСТОЕ УДАЛЕНИЕ через LaMa (достраивает фон) ===
        if not user_prompt:
            with _lock:
                _set_stage("Убираю…")
                mask_l = mask.filter(ImageFilter.MaxFilter(7))   # чуть расширить (убрать ореолы)
                try:
                    result = _lama_remove(init, mask_l)
                except Exception as e:
                    _set_stage("", 0)
                    return jsonify({"error": f"LaMa недоступна: {e}"})
            _set_stage("", 0)
            data, name = _save_and_encode(result)
            return jsonify({"image": data, "file": name})

        # === РЕЖИМ 2: есть описание = ДОРИСОВАТЬ новое через SDXL-инпейнт ===
        prompt = _enhance_prompt(user_prompt, d.get("model", DEFAULT_MODEL)) if d.get("enhance", True) else user_prompt
        neg = _build_negative(d.get("negative"))
        sc = 1024.0 / max(W0, H0)                             # длинная сторона ~1024, кратно 8
        Wp, Hp = _snap(W0 * sc), _snap(H0 * sc)
        init_p = init.resize((Wp, Hp), Image.LANCZOS)
        mask_p = mask.resize((Wp, Hp), Image.NEAREST).filter(ImageFilter.MaxFilter(9))
        with _lock:
            _set_stage("Готовлю модель…")
            pipe = _ensure_inpaint()
            if pipe is None:                                 # спец-модель не скачана — старый способ
                _ensure(d.get("model", DEFAULT_MODEL))
                pipe = _get_inpaint()
            _set_stage("Дорисовываю", 28)
            out = pipe(prompt=prompt, negative_prompt=neg, image=init_p, mask_image=mask_p,
                       width=Wp, height=Hp, num_inference_steps=28, guidance_scale=8.0,
                       strength=1.0, callback_on_step_end=_step_cb).images[0]
        # композитинг: меняем ТОЛЬКО закрашенное, остальное — оригинал в полном разрешении
        out_full = out.resize((W0, H0), Image.LANCZOS)
        mask_soft = mask.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(3))
        result = Image.composite(out_full, init, mask_soft)
        _set_stage("", 0)
        data, name = _save_and_encode(result)
        return jsonify({"image": data, "file": name})
    except Exception as e:
        _set_stage("", 0)
        return jsonify({"error": f"Ошибка: {e}"})

@app.route("/byexample", methods=["POST"])
def byexample():
    """Генерация ПО ОБРАЗЦУ: images — одно или несколько фото-примеров, prompt — что сделать."""
    d = request.json or {}
    prompt = (d.get("prompt") or "").strip()
    images = d.get("images") or []
    if not images:
        return jsonify({"error": "Добавь хотя бы одно фото-образец."})
    model = d.get("model", DEFAULT_MODEL)
    if d.get("enhance", True):
        prompt = _enhance_prompt(prompt or "beautiful high quality image", model)
    scale = max(0.1, min(float(d.get("scale", 0.6)), 1.0))
    try:
        from PIL import Image
        refs = []
        for b in images[:4]:
            if "," in b:
                b = b.split(",", 1)[1]
            refs.append(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB").resize((512, 512)))
        with _lock:
            _ensure(model)
            _load_ipadapter()
            try:
                _txt.set_ip_adapter_scale(scale)
                ip_img = refs if len(refs) > 1 else refs[0]
                img = _txt(prompt=prompt or "high quality, detailed",
                           negative_prompt=BASE_NEGATIVE,
                           ip_adapter_image=ip_img, num_inference_steps=30,
                           guidance_scale=6.0, width=1024, height=1024).images[0]
            finally:
                try:
                    _txt.unload_ip_adapter()   # чтобы обычные режимы работали чисто
                except Exception:
                    pass
        data, name = _save_and_encode(img)
        return jsonify({"image": data, "file": name})
    except Exception as e:
        return jsonify({"error": f"Ошибка: {e}"})

if __name__ == "__main__":
    LOG = os.path.join(BASE, "last_run.log")
    def _log(m):
        print(m, flush=True)
        try:
            open(LOG, "a", encoding="utf-8").write(str(m) + "\n")
        except Exception:
            pass
    try:
        open(LOG, "w", encoding="utf-8").write("")
    except Exception:
        pass
    try:
        _log("=" * 50)
        _log("  Генератор картинок ЗАПУЩЕН!")
        _log("  Открой в браузере:  http://localhost:5001")
        _log("  Первая картинка дольше (грузится модель). Окно не закрывай.")
        _log("=" * 50)
        threading.Thread(target=_ensure_ollama, daemon=True).start()   # для умного промпта
        import webbrowser, time as _t
        def _open():
            import urllib.request
            for _ in range(120):
                try:
                    urllib.request.urlopen("http://localhost:5001/", timeout=1)
                    webbrowser.open("http://localhost:5001"); return
                except Exception:
                    _t.sleep(0.5)
        threading.Thread(target=_open, daemon=True).start()
        app.run(host="127.0.0.1", port=5001, threaded=True)
    except Exception as e:
        import traceback
        _log("!!! ОШИБКА ЗАПУСКА: " + repr(e))
        _log(traceback.format_exc())
        input("Ошибка — покажи боту файл last_run.log. Enter чтобы закрыть...")
