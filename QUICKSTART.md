# Quick Start / Быстрый старт

---

## 1️⃣ EN — Activate virtual environment / RU — Активация виртуального окружения

```bash
# RU: Перейти в папку проекта / EN: Navigate to project folder
cd "/Users/kkolesss/Documents/VSCode Projects/clean-metadata"

# RU: Активировать виртуальное окружение / EN: Activate virtual environment
source venv/bin/activate
```

RU: После активации в терминале появится `(venv)` в начале строки.
EN: After activation you will see `(venv)` at the start of the terminal prompt.

---

## 2️⃣ EN — Verify installation / RU — Проверка установки

```bash
python clean_metadata.py --check-deps
```

**RU: Все зависимости установлены** ✅ / **EN: All dependencies installed** ✅

| Component | Status |
|-----------|--------|
| Pillow    | ✅ |
| numpy     | ✅ |
| scipy     | ✅ |
| PyWavelets | ✅ |
| mutagen   | ✅ |
| exiftool  | ✅ |
| ffmpeg    | ✅ |

---

## 3️⃣ EN — Basic usage / RU — Базовое использование

```bash
# RU: Обработать одно изображение
# EN: Process a single image
python clean_metadata.py examples/your_photo.jpg

# RU: Обработать с подробным выводом
# EN: Process with verbose output
python clean_metadata.py examples/your_photo.jpg --wm-method ensemble --wm-strength 0.5 -v

# RU: Обработать всю папку
# EN: Process entire folder
python clean_metadata.py ./examples -o ./cleaned
```

RU: Результаты сохраняются в папку `./cleaned/`
EN: Results are saved to `./cleaned/`

---

## 4️⃣ EN — External tools / RU — Внешние инструменты

RU: Оба инструмента уже установлены. Команды установки приведены для справки.
EN: Both tools are already installed. Install commands listed for reference.

### ExifTool (RU: полное удаление метаданных / EN: full metadata removal)
```bash
brew install exiftool
```

### FFmpeg (RU: обработка видео / EN: video processing)
```bash
brew install ffmpeg
```

---

## 5️⃣ EN — Deactivate environment / RU — Деактивация окружения

```bash
deactivate
```

---

## EN — Quick command reference / RU — Примеры команд

```bash
# RU: Простая обработка / EN: Simple processing
python clean_metadata.py photo.jpg

# RU: С анализом watermark / EN: With watermark analysis
python clean_metadata.py photo.jpg --analyze -v

# RU: Только метаданные (без watermark-атаки) / EN: Metadata only (no watermark attack)
python clean_metadata.py photo.jpg --no-watermark-attack

# RU: Вся папка с максимальной атакой / EN: Entire folder with maximum attack
python clean_metadata.py ./examples --wm-method ensemble --wm-strength 0.7 -r

# RU: Сравнение метаданных до/после / EN: Metadata comparison before/after
python clean_metadata.py photo.jpg --compare

# RU: Управление качеством JPEG / EN: Control output JPEG quality
python clean_metadata.py photo.jpg --quality 92
```

---

## EN — Project Structure / RU — Структура проекта

```
clean-metadata/
├── venv/                    ← Python virtual environment / Виртуальное окружение
├── clean_metadata.py        ← Main script / Основной скрипт
├── requirements.txt         ← Dependencies / Зависимости
├── requirements-lock.txt    ← Pinned versions / Зафиксированные версии
├── README.md                ← Full docs (RU + EN) / Полная документация
├── QUICKSTART.md            ← This file / Этот файл
├── PROJECT_INFO.md          ← Project status / Статус проекта
├── COMMANDS.txt             ← Command cheatsheet / Шпаргалка
├── examples/                ← Place input files here / Входные файлы
└── cleaned/                 ← Results appear here / Результаты
```

---

## EN — Troubleshooting / RU — Проблемы?

### "command not found: python"
RU: Используйте `python3` вместо `python`
EN: Use `python3` instead of `python`

### RU: Виртуальное окружение не активируется / EN: venv not activating
```bash
source "/Users/kkolesss/Documents/VSCode Projects/clean-metadata/venv/bin/activate"
```

### RU: Забыли активировать venv? / EN: Forgot to activate venv?
RU: Проверьте, есть ли `(venv)` в начале строки терминала. Если нет:
EN: Check for `(venv)` at the start of the terminal prompt. If missing:
```bash
source venv/bin/activate
```

---

## EN — Full documentation / RU — Подробная документация

See **[README.md](README.md)** / См. **[README.md](README.md)**
