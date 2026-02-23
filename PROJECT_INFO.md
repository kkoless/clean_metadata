# Project Info: Clean Metadata / Информация о проекте Clean Metadata

---

## EN — Project Status

**The project is fully configured and ready to use.**

---

## RU — Статус установки

**Проект полностью настроен и готов к использованию.**

---

## 📦 EN — Installed Components / RU — Установленные компоненты

### Python packages (installed in venv) / Python пакеты (установлены в venv)

| Package / Пакет | Version / Версия | Purpose / Назначение |
|-----------------|------------------|----------------------|
| **Pillow**      | 11.3.0 | Image processing / Обработка изображений (обязательно) |
| **numpy**       | 2.0.2  | Numerical ops for watermark attacks / Численные операции (обязательно) |
| **scipy**       | 1.13.1 | DCT and geometric attacks / DCT и геометрические атаки |
| **PyWavelets**  | 1.6.0  | Wavelet-domain attacks (DWT) / Атаки в вейвлет-домене |
| **mutagen**     | 1.47.0 | Audio metadata removal / Удаление метаданных из аудио |

### External tools / Внешние инструменты

| Tool / Инструмент | Status / Статус | Install (macOS) / Установка | Purpose / Назначение |
|-------------------|-----------------|----------------------------|----------------------|
| **exiftool** | ✅ Installed / Установлен | `brew install exiftool` | Full metadata removal / Полная очистка метаданных |
| **ffmpeg**   | ✅ Installed / Установлен | `brew install ffmpeg`   | Video processing / Обработка видео |

---

## 🗂️ Project Structure / Структура проекта

```
clean-metadata/
│
├── venv/                       # ✅ Python 3.9.6 virtual environment
│   ├── bin/                    # Executables (python, pip)
│   └── lib/                    # Installed packages
│
├── clean_metadata.py           # ✅ Main script
├── requirements.txt            # ✅ Dependency list
├── requirements-lock.txt       # ✅ Pinned versions
│
├── README.md                   # ✅ Full documentation (RU + EN)
├── QUICKSTART.md               # ✅ Quick start guide (RU + EN)
├── PROJECT_INFO.md             # ✅ This file (RU + EN)
├── COMMANDS.txt                # ✅ Command cheatsheet (RU + EN)
│
├── .gitignore                  # ✅ Git ignore rules
│
├── examples/                   # 📁 Input files folder
│   └── .gitkeep
│
└── cleaned/                    # 📁 Output files folder (auto-created)
```

---

## 🚀 EN — Getting Started

### 1. Activate virtual environment

```bash
cd "/Users/kkolesss/Documents/VSCode Projects/clean-metadata"
source venv/bin/activate
```

After activation you will see `(venv)` at the start of the terminal line.

### 2. Verify installation

```bash
python clean_metadata.py --check-deps
```

**Expected output:**
```
External tools:
  ✓ ffmpeg       — video
  ✓ exiftool     — full metadata removal

Python packages:
  ✓ Pillow       — images (required)
  ✓ numpy        — watermark attacks (required)
  ✓ scipy        — DCT and geometric attacks
  ✓ PyWavelets   — wavelet attack
  ✓ mutagen      — audio
```

### 3. First run

```bash
python clean_metadata.py examples/photo.jpg
```

Result will appear in `cleaned/photo_clean.jpg`

---

## 🚀 RU — Как начать работу

### 1. Активация виртуального окружения

```bash
cd "/Users/kkolesss/Documents/VSCode Projects/clean-metadata"
source venv/bin/activate
```

После активации появится `(venv)` в начале строки.

### 2. Проверка установки

```bash
python clean_metadata.py --check-deps
```

**Ожидаемый результат:**
```
Внешние инструменты:
  ✓ ffmpeg       — видео
  ✓ exiftool     — полная очистка метаданных

Python пакеты:
  ✓ Pillow       — изображения (обязательно)
  ✓ numpy        — атаки на watermark (обязательно)
  ✓ scipy        — DCT и geometric атаки
  ✓ PyWavelets   — wavelet атака
  ✓ mutagen      — аудио
```

### 3. Первый запуск

```bash
python clean_metadata.py examples/photo.jpg
```

Результат появится в папке `cleaned/photo_clean.jpg`

---

## 📝 EN — Key Commands / RU — Основные команды

```bash
# Simple processing / Простая обработка
python clean_metadata.py photo.jpg

# With detailed analysis / С детальным анализом
python clean_metadata.py photo.jpg --analyze -v

# Process entire folder / Обработка всей папки
python clean_metadata.py ./examples --wm-method ensemble --wm-strength 0.5

# Metadata only, no watermark attack / Только метаданные, без watermark-атаки
python clean_metadata.py photo.jpg --no-watermark-attack

# Control output JPEG quality / Управление качеством JPEG
python clean_metadata.py photo.jpg --quality 92
```

---

## 🔧 EN — Technical Info / RU — Техническая информация

| Parameter / Параметр | Value / Значение |
|----------------------|------------------|
| Python version       | 3.9.6            |
| venv size            | ~30–40 MB        |
| Project path         | `/Users/kkolesss/Documents/VSCode Projects/clean-metadata` |

### Update dependencies / Обновление зависимостей

```bash
source venv/bin/activate
pip install --upgrade -r requirements.txt
pip freeze > requirements-lock.txt
```

---

## 💡 EN — Tips / RU — Советы

1. **Always activate venv** before running the script / **Всегда активируйте venv** перед запуском
2. **Use `--analyze`** to understand attack effectiveness / **Используйте `--analyze`** для оценки эффективности атак
3. **Start with `--wm-strength 0.5`** for quality/effectiveness balance / **Начните с `--wm-strength 0.5`** для баланса
4. **Use `--quality 90`** (default) for output JPEG / **`--quality 90`** по умолчанию для JPEG
5. **Use `ensemble` method** for maximum effectiveness / **Метод `ensemble`** наиболее эффективен

---

## ❓ EN — Troubleshooting / RU — Поддержка

Check the following / Проверьте следующее:

1. Is venv activated? / Активировано ли окружение? → `(venv)` in terminal prompt
2. Are dependencies installed? / Установлены ли зависимости? → `python clean_metadata.py --check-deps`
3. Correct Python in use? / Правильный Python? → `which python` → path inside venv

---

**Last updated / Дата обновления:** 23 February 2026  
**Python version / Версия Python:** 3.9.6  
**Status / Статус:** ✅ Ready to use / Готов к использованию
