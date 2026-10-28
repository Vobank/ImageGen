# Генератор картинок · ImageGen

**[Русский](#русский) · [English](#english)**

> 🌐 **Язык интерфейса: только русский** · **Interface language: Russian only**

Локальный генератор изображений на Stable Diffusion XL для RTX 5080 — текст→картинка, картинка→картинка и инпейнтинг, через веб-интерфейс.

A local image generator powered by Stable Diffusion XL for the RTX 5080 — text-to-image, image-to-image and inpainting, via a web UI.

---

## Русский

### Что это
Веб-приложение, которое генерирует картинки локально на вашей видеокарте (ничего не уходит в облако).

### Возможности
- 🖼 txt2img — картинка по тексту
- 🔁 img2img — переделка существующей картинки
- 🩹 Инпейнтинг — дорисовка/замена части
- ⚙ Настройки качества (шаги, CFG, размеры)

### Требования
- Windows 10/11, видеокарта NVIDIA (8+ ГБ VRAM, лучше больше)
- Python 3.12, torch cu128 (под Blackwell/RTX 50xx)
- Модели SDXL (качаются отдельно)

### Запуск
- `pip install -r requirements.txt`
- `Генератор картинок.bat` — запустит сервер и откроет интерфейс в браузере

---

## English

### What it is
A web app that generates images locally on your GPU (nothing goes to the cloud).

### Features
- 🖼 txt2img — image from a text prompt
- 🔁 img2img — transform an existing image
- 🩹 Inpainting — fill/replace part of an image
- ⚙ Quality settings (steps, CFG, size)

### Requirements
- Windows 10/11, NVIDIA GPU (8+ GB VRAM, more is better)
- Python 3.12, torch cu128 (for Blackwell / RTX 50xx)
- SDXL models (downloaded separately)

### Run
- `pip install -r requirements.txt`
- `Генератор картинок.bat` — starts the server and opens the UI in your browser

---

### Технологии / Tech
Python · Stable Diffusion XL · PyTorch (cu128) · веб-интерфейс / web UI

### ⚠️
Личный проект. / Personal project.
