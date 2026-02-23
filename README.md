# Clean Metadata

**RU:** Инструмент для удаления метаданных и атак на невидимые водяные знаки (SynthID, C2PA, HiDDeN и др.) из изображений, видео и аудио файлов.

**EN:** A tool for removing metadata and attacking invisible watermarks (SynthID, C2PA, HiDDeN, etc.) from images, video and audio files.

**Для научно-исследовательских целей / For research purposes** (media forensics, watermark robustness testing).

---

## RU — Возможности / EN — Features

### RU — Удаление метаданных / EN — Metadata Removal
- **EXIF, XMP, IPTC** data from images / данные из изображений
- **Metadata chunks** from PNG files / чанки из PNG файлов
- **APP segments** from JPEG files / APP-сегменты из JPEG файлов
- **Metadata streams** from video / потоки из видео
- **ID3 tags** from audio files / теги из аудио файлов

### RU — Атаки на невидимые водяные знаки / EN — Invisible Watermark Attacks
- **Gaussian noise injection** (Stirmark classic)
- **JPEG double compression** (frequency-domain attack)
- **DCT-coefficient perturbation** (direct frequency attack)
- **Wavelet-domain noise** (DWT attack)
- **Brightness/contrast jitter** (photometric transforms)
- **Geometric micro-distortion** (sub-pixel warping)
- **Median filter** (watermark pattern smoothing)
- **FGSM-style adversarial noise** (Goodfellow et al., 2014)
- **Combined ensemble attack** (all methods combined) ⭐ **Recommended**

---

## RU — Установка / EN — Installation

### 1. RU — Клонируйте или перейдите в папку проекта / EN — Clone or navigate to the project folder

### 2. RU — Создайте и активируйте виртуальное окружение / EN — Create and activate virtual environment

```bash
# RU: Создание / EN: Create
python3 -m venv venv

# RU: Активация macOS/Linux / EN: Activate macOS/Linux
source venv/bin/activate

# EN: Activate Windows
# venv\Scripts\activate
```

### 3. RU — Установите Python-зависимости / EN — Install Python dependencies

```bash
pip install -r requirements.txt
```

Это установит / This installs:
- **Pillow** — image processing / обработка изображений
- **numpy** — numerical operations / численные операции
- **scipy** — DCT and geometric attacks / DCT и геометрические атаки
- **PyWavelets** — wavelet-domain attacks / атаки в вейвлет-домене
- **mutagen** — audio metadata removal / удаление метаданных аудио

### 4. RU — Внешние инструменты / EN — External tools (optional but recommended)

#### ExifTool (RU: полное удаление метаданных / EN: full metadata removal)
- **macOS**: `brew install exiftool`
- **Linux**: `sudo apt-get install libimage-exiftool-perl`
- **Windows**: [https://exiftool.org](https://exiftool.org)

#### FFmpeg (RU: обработка видео / EN: video processing)
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg`
- **Windows**: [https://ffmpeg.org](https://ffmpeg.org)

---

## RU — Использование / EN — Usage

### RU — Базовое использование / EN — Basic usage

```bash
# RU: Обработать одно изображение (ensemble attack, рекомендуется)
# EN: Process a single image (ensemble attack, recommended)
python clean_metadata.py photo.jpg

# RU: Только удаление метаданных, без атаки на watermark
# EN: Metadata removal only, no watermark attack
python clean_metadata.py photo.jpg --no-watermark-attack
```

### RU — Продвинутое использование / EN — Advanced usage

```bash
# RU: Рекомендуемый режим для исследования
# EN: Recommended mode for research
python clean_metadata.py photo.jpg --wm-method ensemble --wm-strength 0.5 -v

# RU: Вся папка с максимальной атакой
# EN: Entire folder with maximum attack
python clean_metadata.py ./examples -o ./cleaned --wm-method ensemble --wm-strength 0.7 -a -r

# RU: Обработка с анализом watermark до и после
# EN: Process with watermark analysis before and after
python clean_metadata.py photo.jpg --analyze --wm-method ensemble -v

# RU: Сравнение метаданных до/после
# EN: Compare metadata before/after
python clean_metadata.py photo.jpg --compare

# RU: Управление качеством выходного JPEG
# EN: Control output JPEG quality
python clean_metadata.py photo.jpg --quality 92

# RU: Обработка видео с перекодированием
# EN: Process video with re-encoding
python clean_metadata.py video.mp4 -a

# RU: Обработка аудио
# EN: Process audio
python clean_metadata.py track.mp3
```

---

## RU — Опции командной строки / EN — Command-line Options

### RU — Основные параметры / EN — Main Parameters

| Option / Опция | RU — Описание | EN — Description |
|----------------|---------------|------------------|
| `inputs` | Файл или папка для обработки | File or folder to process |
| `-o, --output` | Папка для результатов (по умолчанию: `./cleaned`) | Output folder (default: `./cleaned`) |
| `-v, --verbose` | Подробный вывод процесса | Verbose output |
| `-a, --aggressive` | Агрессивный режим (перекодирование видео) | Aggressive mode (re-encode video, recreate pixels) |
| `-r, --randomize-timestamps` | Рандомизировать timestamps файлов | Randomize file timestamps |
| `-c, --compare` | Сравнить метаданные до/после | Compare metadata before/after (requires exiftool) |
| `--analyze` | Анализировать файлы на watermark | Analyze files for watermark signatures |
| `--quality` | Качество JPEG на выходе (60–97, default=90) | Output JPEG quality (60–97, default=90) |
| `--check-deps` | Проверить установленные зависимости | Check installed dependencies |

### RU — Параметры watermark-атаки / EN — Watermark Attack Parameters

| Option / Опция | RU — Описание | EN — Description |
|----------------|---------------|------------------|
| `--wm-method` | Метод атаки | Attack method |
| `--wm-strength` | Сила атаки 0.0–1.0 (рекомендуется 0.3–0.6) | Attack strength 0.0–1.0 (recommended 0.3–0.6) |
| `--no-watermark-attack` | Только метаданные, без атаки | Metadata only, no watermark attack |

### RU — Методы watermark-атаки / EN — Watermark Attack Methods

| Method | RU | EN |
|--------|----|----|
| **`ensemble`** ⭐ | Все методы вместе (наиболее эффективно) | All methods combined (most effective) |
| **`gaussian`** | Гауссовский шум (Stirmark) | Gaussian noise (Stirmark) |
| **`dct`** | Возмущение DCT-коэффициентов | DCT-coefficient perturbation |
| **`wavelet`** | Шум в вейвлет-домене | Wavelet-domain noise |
| **`photometric`** | Изменения яркости/контраста/гаммы | Brightness/contrast/gamma changes |
| **`geometric`** | Микро-деформации пикселей | Pixel micro-distortion |
| **`fgsm`** | Adversarial noise (FGSM-style) | Adversarial noise (FGSM-style) |
| **`median`** | Медианный фильтр | Median filter |
| **`jpeg`** | Двойное JPEG-сжатие | Double JPEG compression |

---

## RU — Примеры использования / EN — Usage Examples

### Example 1: RU — Быстрая обработка / EN — Quick processing

```bash
python clean_metadata.py examples/photo.jpg
```

**RU — Результат:** Удалены метаданные, применена ensemble-атака, файл в `./cleaned/photo_clean.jpg`

**EN — Result:** Metadata removed, ensemble attack applied, file saved to `./cleaned/photo_clean.jpg`

### Example 2: RU — Обработка папки с анализом / EN — Folder processing with analysis

```bash
python clean_metadata.py ./examples --wm-method ensemble --wm-strength 0.5 --analyze -v
```

**RU:** Все файлы в папке обработаны, проведён анализ watermark до и после, подробный вывод.

**EN:** All supported files in the folder processed, watermark analysis before and after, verbose output.

### Example 3: RU — Только метаданные / EN — Metadata only

```bash
python clean_metadata.py photo.jpg --no-watermark-attack --compare
```

**RU:** Удалены только метаданные, пиксели не изменены, показано сравнение до/после.

**EN:** Metadata removed only, pixels unchanged, before/after comparison shown.

### Example 4: RU — Управление качеством JPEG / EN — JPEG quality control

```bash
# RU: Высокое качество (почти без потерь)
# EN: High quality (near-lossless)
python clean_metadata.py photo.jpg --quality 95

# RU: Компактный файл
# EN: Compact file size
python clean_metadata.py photo.jpg --quality 75
```

---

## RU — Академические ссылки / EN — Academic References

- Fernandez et al. "The Stable Signature" (2023)
- Zhao et al. "Invisible Image Watermarks Are Provably Removable" (2023)
- Saberi et al. "Robustness of AI-Image Detectors" (2023)
- Yang et al. "Gaussian Shading" (2024)
- Petitcolas et al. "Stirmark Benchmark" (1998)
- Goodfellow et al. "FGSM" (2014)
- Corvi et al. "Intriguing Properties of Diffusion Models" (2023)
- Jiang et al. "WEvader" (2023)

---

## RU — Проверка зависимостей / EN — Check Dependencies

```bash
python clean_metadata.py --check-deps
```

---

## RU — Структура проекта / EN — Project Structure

```
clean-metadata/
├── venv/                    # Python virtual environment
├── clean_metadata.py        # Main script
├── requirements.txt         # Python dependencies
├── requirements-lock.txt    # Pinned versions
├── README.md                # This file (RU + EN)
├── QUICKSTART.md            # Quick start guide (RU + EN)
├── PROJECT_INFO.md          # Project info (RU + EN)
├── COMMANDS.txt             # Command cheatsheet (RU + EN)
├── .gitignore
├── examples/                # Input files folder
└── cleaned/                 # Output files folder (auto-created)
```

---

## RU — Важные примечания / EN — Important Notes

1. **RU:** Качество изображения: при `wm-strength > 0.7` возможна заметная деградация
   **EN:** Image quality: at `wm-strength > 0.7` visible degradation is possible

2. **PSNR:**
   - `> 42 dB` — отличное / excellent (invisible)
   - `> 36 dB` — хорошее / good
   - `> 30 dB` — приемлемое / acceptable
   - `< 30 dB` — заметная деградация / visible degradation

3. **RU — Поддерживаемые форматы / EN — Supported formats:**
   - **Images / Изображения:** JPG, PNG, GIF, TIFF, BMP, WebP, AVIF, HEIC
   - **Video / Видео:** MP4, MOV, AVI, MKV, WebM, FLV, WMV
   - **Audio / Аудио:** MP3, FLAC, OGG, M4A, WAV, AIFF, WMA

4. **Ethical Use / Этика:** Research and watermarking robustness testing only / Только для научных исследований

---

## RU — Устранение неполадок / EN — Troubleshooting

### "command not found: python"
Use `python3` instead / Используйте `python3` вместо `python`

### ExifTool not found / ExifTool не найден
```bash
# macOS
brew install exiftool
# Linux
sudo apt-get install libimage-exiftool-perl
```

### FFmpeg not found / FFmpeg не найден
```bash
# macOS
brew install ffmpeg
# Linux
sudo apt-get install ffmpeg
```

---

## RU — Лицензия / EN — License

For scientific research and educational purposes only.
Только для научно-исследовательских и образовательных целей.
