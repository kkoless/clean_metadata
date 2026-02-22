#!/usr/bin/env python3
"""
Удаление метаданных + атаки на невидимые водяные знаки (SynthID и аналоги).
Для научно-исследовательских целей (медиафорензика, watermark robustness).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ЧТО ДЕЛАЕТ СКРИПТ:
   1. Удаляет все метаданные (EXIF/XMP/IPTC/chunks)
   2. Атакует невидимые водяные знаки (SynthID, C2PA, HiDDeN, и др.)
      через набор техник из академической литературы

 ТЕХНИКИ АТАК НА WATERMARK:
   - Gaussian noise injection          (классика, Stirmark)
   - JPEG double compression           (атака на частотный домен)
   - DCT-coefficient perturbation      (прямая атака на частоты)
   - Wavelet-domain noise              (атака через DWT)
   - Brightness/contrast jitter        (фотометрические трансформации)
   - Geometric micro-distortion        (субпиксельные деформации)
   - Median filter                     (сглаживание watermark-паттернов)
   - FGSM-style adversarial noise      (Goodfellow et al., 2014)
   - Combined ensemble attack          (все методы вместе)

 ССЫЛКИ:
   Fernandez et al. "The Stable Signature" (2023)
   Zhao et al. "Invisible Image Watermarks Are Provably Removable" (2023)
   Saberi et al. "Robustness of AI-Image Detectors" (2023)
   Yang et al. "Gaussian Shading" (2024)

 ЗАВИСИМОСТИ:
   pip install Pillow numpy scipy pywavelets
   pip install torch torchvision   # опционально, для FGSM
   ffmpeg  — https://ffmpeg.org
   exiftool — https://exiftool.org
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import shutil
import struct
import subprocess
import argparse
import random
import json
import math
import tempfile
from pathlib import Path
from datetime import datetime
from io import BytesIO

import numpy as np

try:
    from PIL import Image, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[!] Установите Pillow: pip install Pillow")

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import pywt
    HAS_WAVELETS = True
except ImportError:
    HAS_WAVELETS = False


# ══════════════════════════════════════════════════════════════
#  УТИЛИТЫ
# ══════════════════════════════════════════════════════════════

def run(cmd, capture=True):
    r = subprocess.run(cmd, capture_output=capture, text=True)
    return r.returncode == 0, r.stderr


def has_tool(name):
    return shutil.which(name) is not None


def randomize_timestamps(path: Path):
    ts = random.uniform(
        datetime(2020, 1, 1).timestamp(),
        datetime(2024, 12, 31).timestamp()
    )
    os.utime(path, (ts, ts))


def img_to_array(img: "Image.Image") -> np.ndarray:
    return np.array(img, dtype=np.float32)


def array_to_img(arr: np.ndarray, mode="RGB") -> "Image.Image":
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode)


def psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio — мера качества после атаки."""
    mse = np.mean((original - modified) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


# ══════════════════════════════════════════════════════════════
#  АТАКИ НА НЕВИДИМЫЕ ВОДЯНЫЕ ЗНАКИ
# ══════════════════════════════════════════════════════════════

class WatermarkAttacker:
    """
    Набор атак на invisible watermarks (SynthID, HiDDeN, RivaGAN, TrustMark и др.)
    Каждая атака минимально деградирует визуальное качество, но нарушает
    статистические паттерны, в которых закодирован водяной знак.
    """

    def __init__(self, strength: float = 0.5, verbose: bool = False):
        """
        strength: 0.0–1.0, где 0.0 = минимальное воздействие, 1.0 = максимальное.
        Рекомендуется 0.3–0.6 для баланса качество/эффективность.
        """
        self.strength = strength
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"    [wm] {msg}")

    # ──────────────────────────────────────────────────────────
    # 1. Гауссовский шум (Stirmark-style)
    # ──────────────────────────────────────────────────────────
    def gaussian_noise(self, arr: np.ndarray) -> np.ndarray:
        """
        Добавляет Гауссовский шум с σ пропорциональным strength.
        Нарушает высокочастотные паттерны watermark без видимой деградации.
        Ref: Stirmark Benchmark (Petitcolas et al., 1998)
        """
        sigma = self.strength * 8.0  # σ ∈ [0, 8] при strength ∈ [0, 1]
        noise = np.random.normal(0, sigma, arr.shape)
        self._log(f"Gaussian noise σ={sigma:.2f}")
        return arr + noise

    # ──────────────────────────────────────────────────────────
    # 2. Атака через двойное JPEG-сжатие
    # ──────────────────────────────────────────────────────────
    def double_jpeg(self, img: "Image.Image") -> "Image.Image":
        """
        Двойное JPEG-сжатие с разными quality factor'ами.
        DCT-квантизация разрушает тонкие частотные паттерны watermark.
        Ref: Кlassic атака, описана в Barni et al. (2001)
        """
        q1 = int(85 - self.strength * 20)  # 65–85
        q2 = int(90 - self.strength * 10)  # 80–90

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q1, subsampling=0)
        buf.seek(0)
        img2 = Image.open(buf).copy()

        buf2 = BytesIO()
        img2.save(buf2, format="JPEG", quality=q2, subsampling=0)
        buf2.seek(0)
        result = Image.open(buf2).copy()

        self._log(f"Double JPEG: q1={q1}, q2={q2}")
        return result

    # ──────────────────────────────────────────────────────────
    # 3. Прямая атака на DCT-коэффициенты
    # ──────────────────────────────────────────────────────────
    def dct_perturbation(self, arr: np.ndarray) -> np.ndarray:
        """
        Вносит возмущения в DCT-коэффициенты средних частот блоками 8×8.
        Атакует именно те частоты, где обычно встраивается watermark.
        Ref: Cox et al. "Watermarking as communications with side information" (1999)
        """
        result = arr.copy()
        h, w = arr.shape[:2]
        epsilon = self.strength * 3.0

        for c in range(arr.shape[2]):
            for y in range(0, h - 8, 8):
                for x in range(0, w - 8, 8):
                    block = result[y:y+8, x:x+8, c]
                    # DCT блока
                    dct_block = self._dct2(block)
                    # Возмущаем средние частоты (зона watermark)
                    perturbation = np.random.uniform(-epsilon, epsilon, (8, 8))
                    # Маска средних частот (избегаем DC и высокие частоты)
                    mask = np.zeros((8, 8))
                    mask[1:5, 1:5] = 1.0
                    dct_block += perturbation * mask
                    # Обратное DCT
                    result[y:y+8, x:x+8, c] = self._idct2(dct_block)

        self._log(f"DCT perturbation ε={epsilon:.2f}")
        return result

    def _dct2(self, block: np.ndarray) -> np.ndarray:
        """2D DCT через разделимые 1D DCT."""
        from scipy.fft import dct
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def _idct2(self, block: np.ndarray) -> np.ndarray:
        """2D обратное DCT."""
        from scipy.fft import idct
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    # ──────────────────────────────────────────────────────────
    # 4. Атака через вейвлет-преобразование (DWT)
    # ──────────────────────────────────────────────────────────
    def wavelet_noise(self, arr: np.ndarray) -> np.ndarray:
        """
        Вносит шум в высокочастотные вейвлет-коэффициенты (детали).
        SynthID и аналоги часто кодируют информацию в HH/HL/LH субдиапазонах.
        Ref: Jiang et al. "WEvader" (2023)
        """
        if not HAS_WAVELETS:
            self._log("pywt не установлен, пропуск DWT-атаки")
            return arr

        result = arr.copy()
        epsilon = self.strength * 5.0

        for c in range(arr.shape[2]):
            coeffs = pywt.dwt2(result[:, :, c], 'db4')
            cA, (cH, cV, cD) = coeffs

            # Атакуем детальные коэффициенты (высокие частоты)
            cH += np.random.normal(0, epsilon, cH.shape)
            cV += np.random.normal(0, epsilon, cV.shape)
            cD += np.random.normal(0, epsilon, cD.shape)

            result[:, :, c] = pywt.idwt2((cA, (cH, cV, cD)), 'db4')

        self._log(f"Wavelet noise ε={epsilon:.2f}")
        return result

    # ──────────────────────────────────────────────────────────
    # 5. Фотометрические трансформации
    # ──────────────────────────────────────────────────────────
    def photometric_jitter(self, arr: np.ndarray) -> np.ndarray:
        """
        Случайные изменения яркости, контраста и гаммы.
        Нарушают абсолютные значения пикселей, в которых закодирован WM.
        """
        result = arr.copy()

        # Гамма-коррекция
        gamma = 1.0 + (random.random() - 0.5) * self.strength * 0.3
        result = np.power(result / 255.0, gamma) * 255.0

        # Яркость
        brightness = (random.random() - 0.5) * self.strength * 15
        result += brightness

        # Контраст
        contrast = 1.0 + (random.random() - 0.5) * self.strength * 0.2
        mean = np.mean(result)
        result = (result - mean) * contrast + mean

        self._log(f"Photometric jitter: γ={gamma:.3f}, b={brightness:.1f}, c={contrast:.3f}")
        return result

    # ──────────────────────────────────────────────────────────
    # 6. Геометрические микро-деформации
    # ──────────────────────────────────────────────────────────
    def geometric_distortion(self, arr: np.ndarray) -> np.ndarray:
        """
        Субпиксельные случайные деформации (displacement field).
        Нарушают пространственные зависимости, на которых основан WM.
        Ref: Stirmark random bend (Petitcolas et al., 1998)
        """
        if not HAS_SCIPY:
            return arr

        h, w = arr.shape[:2]
        amplitude = self.strength * 1.5  # в пикселях

        # Случайное поле смещений
        dy = ndimage.gaussian_filter(
            np.random.randn(h, w) * amplitude, sigma=3
        )
        dx = ndimage.gaussian_filter(
            np.random.randn(h, w) * amplitude, sigma=3
        )

        y_coords, x_coords = np.mgrid[0:h, 0:w]
        y_new = np.clip(y_coords + dy, 0, h - 1).astype(np.float32)
        x_new = np.clip(x_coords + dx, 0, w - 1).astype(np.float32)

        result = np.zeros_like(arr)
        for c in range(arr.shape[2]):
            result[:, :, c] = ndimage.map_coordinates(
                arr[:, :, c], [y_new, x_new], order=1, mode='reflect'
            )

        self._log(f"Geometric distortion amplitude={amplitude:.2f}px")
        return result

    # ──────────────────────────────────────────────────────────
    # 7. Медианный фильтр
    # ──────────────────────────────────────────────────────────
    def median_filter(self, img: "Image.Image") -> "Image.Image":
        """
        Нелинейный фильтр, уничтожает тонкие пиксельные паттерны.
        Эффективен против LSB-based и pixel-level watermarks.
        """
        size = 3 if self.strength < 0.5 else 5
        self._log(f"Median filter size={size}×{size}")
        return img.filter(ImageFilter.MedianFilter(size=size))

    # ──────────────────────────────────────────────────────────
    # 8. FGSM-style adversarial noise
    # ──────────────────────────────────────────────────────────
    def fgsm_noise(self, arr: np.ndarray) -> np.ndarray:
        """
        Fast Gradient Sign Method (Goodfellow et al., 2014) без модели.
        Аппроксимация: структурированный шум на основе знака градиента
        локальной вариации — нарушает детектор без значимого PSNR-снижения.
        Ref: Zhao et al. "Invisible Image Watermarks Are Provably Removable" (2023)
        """
        epsilon = self.strength * 4.0

        # Аппроксимация "градиента" через локальную вариацию (Sobel)
        result = arr.copy()
        for c in range(arr.shape[2]):
            # Sobel-аппроксимация градиента
            gy = np.gradient(arr[:, :, c], axis=0)
            gx = np.gradient(arr[:, :, c], axis=1)
            grad = gy + gx
            # Добавляем знак градиента (FGSM)
            result[:, :, c] += epsilon * np.sign(grad)

        self._log(f"FGSM-style adversarial noise ε={epsilon:.2f}")
        return result

    # ──────────────────────────────────────────────────────────
    # 9. Ensemble-атака (все методы последовательно)
    # ──────────────────────────────────────────────────────────
    def ensemble_attack(self, img: "Image.Image") -> "Image.Image":
        """
        Последовательное применение всех атак с пониженным strength.
        Наиболее эффективная стратегия для обхода robustness watermarks.
        Ref: Saberi et al. "Robustness of AI-Image Detectors" (2023)
        """
        self._log("=== ENSEMBLE ATTACK ===")
        original_strength = self.strength
        # Снижаем силу каждой атаки, т.к. они применяются совместно
        self.strength = original_strength * 0.4

        arr = img_to_array(img)
        original_arr = arr.copy()

        # Шаг 1: Фотометрические трансформации
        arr = self.photometric_jitter(arr)

        # Шаг 2: Вейвлет-шум
        if HAS_WAVELETS:
            arr = self.wavelet_noise(arr)

        # Шаг 3: Геометрические деформации
        if HAS_SCIPY:
            arr = self.geometric_distortion(arr)

        # Шаг 4: FGSM-шум
        arr = self.fgsm_noise(arr)

        # Шаг 5: Гауссовский шум
        arr = self.gaussian_noise(arr)

        img_result = array_to_img(arr, img.mode)

        # Шаг 6: Медианный фильтр
        img_result = self.median_filter(img_result)

        # Шаг 7: DCT-возмущения (если scipy доступен)
        if HAS_SCIPY:
            arr2 = img_to_array(img_result)
            arr2 = self.dct_perturbation(arr2)
            img_result = array_to_img(arr2, img.mode)

        # Шаг 8: Двойное JPEG (финальное)
        img_result = self.double_jpeg(img_result)

        # Метрика качества
        final_arr = img_to_array(img_result)
        score = psnr(original_arr, final_arr)
        self._log(f"PSNR после ensemble: {score:.2f} dB "
                  f"({'отличное' if score > 40 else 'хорошее' if score > 35 else 'заметное'} качество)")

        self.strength = original_strength
        return img_result

    # ──────────────────────────────────────────────────────────
    # Применить выбранную атаку
    # ──────────────────────────────────────────────────────────
    def apply(self, img: "Image.Image", method: str) -> "Image.Image":
        """Применяет указанный метод к изображению."""
        if img.mode != "RGB":
            img = img.convert("RGB")

        if method == "ensemble":
            return self.ensemble_attack(img)

        arr = img_to_array(img)

        if method == "gaussian":
            arr = self.gaussian_noise(arr)
        elif method == "dct":
            if HAS_SCIPY:
                arr = self.dct_perturbation(arr)
            else:
                print("    [!] scipy нужен для DCT-атаки: pip install scipy")
                return img
        elif method == "wavelet":
            arr = self.wavelet_noise(arr)
        elif method == "photometric":
            arr = self.photometric_jitter(arr)
        elif method == "geometric":
            if HAS_SCIPY:
                arr = self.geometric_distortion(arr)
            else:
                print("    [!] scipy нужен для geometric-атаки: pip install scipy")
                return img
        elif method == "fgsm":
            arr = self.fgsm_noise(arr)
        else:
            print(f"    [!] Неизвестный метод: {method}")
            return img

        result_img = array_to_img(arr, "RGB")

        if method == "median":
            result_img = self.median_filter(result_img)
        if method in ("jpeg", "double_jpeg"):
            result_img = self.double_jpeg(result_img)

        return result_img


# ══════════════════════════════════════════════════════════════
#  УДАЛЕНИЕ МЕТАДАННЫХ (JPEG / PNG низкий уровень)
# ══════════════════════════════════════════════════════════════

def strip_jpeg(input_path: Path, output_path: Path) -> bool:
    with open(input_path, "rb") as f:
        data = f.read()
    if data[:2] != b"\xff\xd8":
        return False

    output = BytesIO()
    output.write(b"\xff\xd8")
    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            break
        marker = data[i:i+2]
        i += 2
        if marker in (b"\xff\xd9", b"\xff\xd8"):
            output.write(marker)
            continue
        if marker == b"\xff\xda":
            output.write(marker)
            output.write(data[i:])
            break
        if i + 2 > len(data):
            break
        length = struct.unpack(">H", data[i:i+2])[0]
        segment_data = data[i:i + length]
        i += length
        if b"\xff\xe0" <= marker <= b"\xff\xef":
            continue
        if marker == b"\xff\xfe":
            continue
        output.write(marker)
        output.write(segment_data)

    with open(output_path, "wb") as f:
        f.write(output.getvalue())
    return True


def strip_png(input_path: Path, output_path: Path) -> bool:
    SKIP = {b"tEXt", b"iTXt", b"zTXt", b"tIME", b"eXIf",
            b"iCCP", b"cHRM", b"gAMA", b"sRGB", b"sBIT",
            b"bKGD", b"hIST", b"sPLT", b"pHYs"}
    PNG_SIG = b"\x89PNG\r\n\x1a\n"

    with open(input_path, "rb") as f:
        data = f.read()
    if data[:8] != PNG_SIG:
        return False

    output = BytesIO()
    output.write(PNG_SIG)
    i = 8
    while i < len(data):
        if i + 12 > len(data):
            break
        length = struct.unpack(">I", data[i:i+4])[0]
        chunk_type = data[i+4:i+8]
        chunk_end = i + 12 + length
        if chunk_type not in SKIP:
            output.write(data[i:chunk_end])
        i = chunk_end

    with open(output_path, "wb") as f:
        f.write(output.getvalue())
    return True


# ══════════════════════════════════════════════════════════════
#  ОБРАБОТКА ИЗОБРАЖЕНИЯ (метаданные + watermark)
# ══════════════════════════════════════════════════════════════

def process_image(
    input_path: Path,
    output_path: Path,
    wm_method: str = "ensemble",
    wm_strength: float = 0.5,
    aggressive_meta: bool = False,
    verbose: bool = False
) -> bool:
    if not HAS_PIL:
        return False

    ext = input_path.suffix.lower()
    print(f"    Шаг 1: Удаление метаданных...")

    # --- Шаг 1: Удаление метаданных ---
    meta_ok = False

    if has_tool("exiftool"):
        ok, _ = run(["exiftool", "-all=", "-overwrite_original",
                     "-o", str(output_path), str(input_path)])
        if ok:
            print(f"    [exiftool] Все теги удалены")
            meta_ok = True

    if not meta_ok:
        if ext in (".jpg", ".jpeg"):
            meta_ok = strip_jpeg(input_path, output_path)
            if meta_ok:
                print(f"    [jpeg-strip] APP-сегменты удалены")
        elif ext == ".png":
            meta_ok = strip_png(input_path, output_path)
            if meta_ok:
                print(f"    [png-strip] Metadata-чанки удалены")

    if not meta_ok:
        # Pillow fallback
        with Image.open(input_path) as img:
            fmt = (img.format or ext.lstrip(".")).upper()
            if fmt == "JPG":
                fmt = "JPEG"
            img.convert("RGB").save(output_path, format=fmt, quality=95)
        print(f"    [pillow] Метаданные удалены при пересохранении")
        meta_ok = True

    # --- Шаг 2: Атака на watermark ---
    print(f"    Шаг 2: Атака на watermark (метод={wm_method}, strength={wm_strength})...")

    attacker = WatermarkAttacker(strength=wm_strength, verbose=verbose)

    with Image.open(output_path) as img:
        img_copy = img.copy()

    original_arr = img_to_array(img_copy.convert("RGB"))

    attacked_img = attacker.apply(img_copy, wm_method)

    # Сохраняем результат
    save_path = output_path
    fmt = ext.lstrip(".").upper()
    if fmt == "JPG":
        fmt = "JPEG"

    save_kw = {}
    if fmt == "JPEG":
        save_kw = {"quality": 95, "optimize": True, "subsampling": 0}
    elif fmt == "PNG":
        save_kw = {"optimize": True}

    attacked_img.save(save_path, format=fmt, **save_kw)

    # Финальный проход ExifTool (атака могла добавить теги)
    if has_tool("exiftool"):
        run(["exiftool", "-all=", "-overwrite_original", str(save_path)])

    # Отчёт о качестве
    final_arr = img_to_array(Image.open(save_path).convert("RGB"))
    score = psnr(original_arr, final_arr)
    quality_label = (
        "отличное (незаметно)" if score > 42 else
        "хорошее"             if score > 36 else
        "приемлемое"          if score > 30 else
        "заметная деградация"
    )
    print(f"    PSNR: {score:.1f} dB — {quality_label}")

    return True


# ══════════════════════════════════════════════════════════════
#  ВИДЕО
# ══════════════════════════════════════════════════════════════

def process_video(input_path: Path, output_path: Path, aggressive: bool = False) -> bool:
    if not has_tool("ffmpeg"):
        print("    [!] ffmpeg не найден")
        return False

    meta_clear = [
        "-metadata", "title=", "-metadata", "comment=",
        "-metadata", "description=", "-metadata", "creation_time=",
        "-metadata", "encoder=", "-metadata", "software=",
        "-metadata", "artist=", "-metadata", "copyright=",
        "-metadata:s", "handler_name=", "-metadata:s", "vendor_id=",
    ]
    base = ["-map_metadata", "-1", "-map_chapters", "-1",
            *meta_clear, "-fflags", "+bitexact",
            "-flags:v", "+bitexact", "-flags:a", "+bitexact"]

    if aggressive:
        cmd = ["ffmpeg", "-y", "-i", str(input_path), *base,
               "-c:v", "libx264", "-crf", "18", "-preset", "slow",
               "-c:a", "aac", "-b:a", "192k", str(output_path)]
        print("    [ffmpeg] Перекодирование + очистка метаданных...")
    else:
        cmd = ["ffmpeg", "-y", "-i", str(input_path), *base,
               "-c", "copy", str(output_path)]
        print("    [ffmpeg] Stream copy + очистка метаданных...")

    ok, err = run(cmd)
    if not ok:
        print(f"    [!] ffmpeg: {err[-300:]}")
        return False

    if has_tool("exiftool"):
        run(["exiftool", "-all=", "-overwrite_original", str(output_path)])
    return True


# ══════════════════════════════════════════════════════════════
#  АУДИО
# ══════════════════════════════════════════════════════════════

def process_audio(input_path: Path, output_path: Path) -> bool:
    try:
        from mutagen import File as MutagenFile
        shutil.copy2(input_path, output_path)
        audio = MutagenFile(output_path)
        if audio:
            audio.delete()
            audio.save()
            print("    [mutagen] Аудио-теги удалены")
            return True
    except ImportError:
        print("    [!] pip install mutagen")
    return False


# ══════════════════════════════════════════════════════════════
#  СРАВНЕНИЕ МЕТАДАННЫХ
# ══════════════════════════════════════════════════════════════

def compare_metadata(original: Path, cleaned: Path):
    if not has_tool("exiftool"):
        return
    TECHNICAL = {
        "SourceFile", "ExifToolVersion", "FileName", "Directory",
        "FileSize", "FileModifyDate", "FileAccessDate", "FileCreateDate",
        "FilePermissions", "FileType", "FileTypeExtension", "MIMEType",
        "ImageWidth", "ImageHeight", "ImageSize", "Megapixels",
        "BitDepth", "ColorType", "Compression", "Filter", "Interlace",
        "EncodingProcess", "BitsPerSample", "ColorComponents",
        "YCbCrSubSampling", "Duration", "AvgBitrate",
    }

    def get_tags(p):
        r = subprocess.run(["exiftool", "-j", str(p)],
                           capture_output=True, text=True)
        try:
            return json.loads(r.stdout)[0] if r.stdout else {}
        except Exception:
            return {}

    before = get_tags(original)
    after = get_tags(cleaned)
    removed = set(before.keys()) - set(after.keys())
    remaining = {k: v for k, v in after.items() if k not in TECHNICAL}

    print(f"\n  📊 Метаданные:")
    print(f"     До: {len(before)} тегов  →  После: {len(after)} тегов  "
          f"(удалено: {len(removed)})")

    if remaining:
        print(f"  ⚠️  Осталось: {', '.join(sorted(remaining.keys()))}")
    else:
        print(f"  ✓  Метаданные полностью удалены")


# ══════════════════════════════════════════════════════════════
#  ПРОВЕРКА ЗАВИСИМОСТЕЙ
# ══════════════════════════════════════════════════════════════

def check_deps():
    print("Зависимости:\n")
    print("  Внешние инструменты:")
    for tool, desc in [("ffmpeg", "видео"), ("exiftool", "полная очистка метаданных")]:
        st = "✓" if has_tool(tool) else "✗"
        print(f"    {st} {tool:<12} — {desc}")

    print("\n  Python пакеты:")
    pkgs = [
        ("PIL",      "Pillow",   "изображения (обязательно)"),
        ("numpy",    "numpy",    "атаки на watermark (обязательно)"),
        ("scipy",    "scipy",    "DCT и geometric атаки"),
        ("pywt",     "PyWavelets","wavelet атака"),
        ("mutagen",  "mutagen",  "аудио"),
    ]
    for mod, pkg, desc in pkgs:
        try:
            __import__(mod)
            print(f"    ✓ {pkg:<12} — {desc}")
        except ImportError:
            print(f"    ✗ {pkg:<12} — pip install {pkg}  ({desc})")
    print()


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
#  САМОСТОЯТЕЛЬНАЯ ПРОВЕРКА WATERMARK / AI-ДЕТЕКЦИЯ
# ══════════════════════════════════════════════════════════════

class WatermarkAnalyzer:
    """
    Локальный анализ изображения на признаки invisible watermark и AI-генерации.
    Методы из академической литературы по steganalysis и media forensics.
    """

    def analyze(self, path: Path, reference_path: Path = None) -> dict:
        """
        Полный анализ файла.
        reference_path — оригинал до обработки (для сравнения).
        """
        if not HAS_PIL:
            print("[!] Нужен Pillow")
            return {}

        with Image.open(path) as img:
            img = img.convert("RGB")
            arr = img_to_array(img)

        results = {}
        print(f"\n{'━'*55}")
        print(f"  АНАЛИЗ: {path.name}")
        print(f"{'━'*55}")

        results["dct_anomaly"]    = self._check_dct_anomaly(arr)
        results["lsb_entropy"]    = self._check_lsb_entropy(arr)
        results["noise_floor"]    = self._check_noise_floor(arr)
        results["frequency_dist"] = self._check_frequency_distribution(arr)
        results["pixel_stats"]    = self._check_pixel_statistics(arr)

        if reference_path and reference_path.exists():
            with Image.open(reference_path) as ref:
                ref_arr = img_to_array(ref.convert("RGB"))
            results["diff_analysis"] = self._diff_analysis(arr, ref_arr)

        self._print_report(results)
        return results

    # ──────────────────────────────────────────────────────────
    # 1. DCT-аномалии (метод из JPEG steganalysis)
    # ──────────────────────────────────────────────────────────
    def _check_dct_anomaly(self, arr: np.ndarray) -> dict:
        """
        Анализирует распределение DCT-коэффициентов блоков 8×8.
        Watermarks искажают естественное распределение коэффициентов.
        """
        if not HAS_SCIPY:
            return {"available": False}

        from scipy.fft import dct as scipy_dct

        h, w = arr.shape[:2]
        channel = arr[:, :, 0]  # анализируем Y-канал (яркость)
        coeffs = []

        for y in range(0, h - 8, 8):
            for x in range(0, w - 8, 8):
                block = channel[y:y+8, x:x+8]
                dct_block = scipy_dct(scipy_dct(block.T, norm='ortho').T, norm='ortho')
                # Средние частоты (зона watermark)
                coeffs.extend(dct_block[1:4, 1:4].flatten().tolist())

        coeffs = np.array(coeffs)

        # Статистики распределения
        mean = float(np.mean(coeffs))
        std  = float(np.std(coeffs))
        kurt = float(self._kurtosis(coeffs))

        # Естественные изображения имеют характерный эксцесс ~3-6
        # Watermarked изображения часто показывают аномальный эксцесс
        anomaly = abs(kurt - 4.5) > 3.0

        return {
            "mean": round(mean, 4),
            "std":  round(std, 4),
            "kurtosis": round(kurt, 4),
            "anomaly_detected": anomaly,
            "label": "⚠️  DCT-аномалия (возможен watermark)" if anomaly else "✓  DCT норма"
        }

    # ──────────────────────────────────────────────────────────
    # 2. LSB-энтропия (анализ младших битов)
    # ──────────────────────────────────────────────────────────
    def _check_lsb_entropy(self, arr: np.ndarray) -> dict:
        """
        LSB (least significant bits) — классический канал стеганографии.
        SynthID использует варианты LSB-embedding в пространственном домене.
        Высокая энтропия LSB → признак встроенных данных.
        """
        lsb = (arr.astype(np.uint8) & 1).flatten()

        # Энтропия Шеннона
        p1 = np.mean(lsb)
        p0 = 1.0 - p1
        if p0 > 0 and p1 > 0:
            entropy = -(p0 * np.log2(p0) + p1 * np.log2(p1))
        else:
            entropy = 0.0

        # Максимальная энтропия = 1.0 (равномерное распределение)
        # Естественные изображения: 0.85–0.98
        # LSB-watermarked: близко к 1.0 (слишком равномерно)
        suspicious = entropy > 0.990

        return {
            "entropy": round(float(entropy), 6),
            "p1_ratio": round(float(p1), 4),
            "suspicious": suspicious,
            "label": "⚠️  LSB-энтропия аномально высокая" if suspicious else "✓  LSB норма"
        }

    # ──────────────────────────────────────────────────────────
    # 3. Анализ шумового пола
    # ──────────────────────────────────────────────────────────
    def _check_noise_floor(self, arr: np.ndarray) -> dict:
        """
        ИИ-изображения имеют характерный шумовой профиль отличный от камер.
        Diffusion-модели создают "слишком чистый" или структурированный шум.
        """
        if not HAS_SCIPY:
            return {"available": False}

        # Высокочастотный остаток (убираем низкочастотную составляющую)
        blurred = ndimage.gaussian_filter(arr, sigma=2)
        residual = arr - blurred

        noise_std  = float(np.std(residual))
        noise_mean = float(np.mean(np.abs(residual)))

        # SNR
        signal_std = float(np.std(arr))
        snr = signal_std / (noise_std + 1e-10)

        # Проверка на структурированность шума (FFT)
        fft = np.fft.fft2(residual[:, :, 0])
        fft_magnitude = np.abs(np.fft.fftshift(fft))
        # Нормализованный пик — структурированный шум даст высокий пик
        fft_peak_ratio = float(np.max(fft_magnitude) / (np.mean(fft_magnitude) + 1e-10))

        structured = fft_peak_ratio > 50  # эмпирический порог

        return {
            "noise_std": round(noise_std, 4),
            "snr": round(snr, 4),
            "fft_peak_ratio": round(fft_peak_ratio, 2),
            "structured_noise": structured,
            "label": "⚠️  Структурированный шум (признак WM)" if structured else "✓  Шум естественный"
        }

    # ──────────────────────────────────────────────────────────
    # 4. Анализ частотного распределения
    # ──────────────────────────────────────────────────────────
    def _check_frequency_distribution(self, arr: np.ndarray) -> dict:
        """
        Анализ спектра изображения.
        ИИ-генераторы оставляют характерные паттерны в частотном домене.
        Ref: Corvi et al. "Intriguing Properties of Diffusion Models" (2023)
        """
        fft2d = np.fft.fft2(arr[:, :, 0])
        magnitude = np.abs(np.fft.fftshift(fft2d))
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Энергия в разных частотных зонах
        def ring_energy(r_min, r_max):
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((y - cy)**2 + (x - cx)**2)
            mask = (dist >= r_min) & (dist < r_max)
            return float(np.sum(magnitude[mask]**2))

        low  = ring_energy(0,  min(cy, cx) * 0.1)
        mid  = ring_energy(min(cy, cx) * 0.1, min(cy, cx) * 0.4)
        high = ring_energy(min(cy, cx) * 0.4, min(cy, cx))

        total = low + mid + high + 1e-10
        low_r  = low  / total
        mid_r  = mid  / total
        high_r = high / total

        return {
            "low_freq_ratio":  round(low_r, 4),
            "mid_freq_ratio":  round(mid_r, 4),
            "high_freq_ratio": round(high_r, 4),
            "label": f"Спектр: low={low_r:.1%} mid={mid_r:.1%} high={high_r:.1%}"
        }

    # ──────────────────────────────────────────────────────────
    # 5. Базовая статистика пикселей
    # ──────────────────────────────────────────────────────────
    def _check_pixel_statistics(self, arr: np.ndarray) -> dict:
        return {
            "mean":   round(float(np.mean(arr)), 2),
            "std":    round(float(np.std(arr)), 2),
            "min":    int(np.min(arr)),
            "max":    int(np.max(arr)),
            "label":  f"μ={np.mean(arr):.1f} σ={np.std(arr):.1f}"
        }

    # ──────────────────────────────────────────────────────────
    # 6. Попиксельное сравнение (если есть оригинал)
    # ──────────────────────────────────────────────────────────
    def _diff_analysis(self, arr: np.ndarray, ref: np.ndarray) -> dict:
        if arr.shape != ref.shape:
            return {"error": "Разный размер изображений"}

        diff = arr.astype(float) - ref.astype(float)
        psnr_val = psnr(ref, arr)
        max_diff = float(np.max(np.abs(diff)))
        mean_diff = float(np.mean(np.abs(diff)))

        return {
            "psnr_db":   round(psnr_val, 2),
            "max_diff":  round(max_diff, 2),
            "mean_diff": round(mean_diff, 4),
            "label": f"PSNR={psnr_val:.1f} dB, макс.откл.={max_diff:.1f}, ср.откл.={mean_diff:.4f}"
        }

    def _kurtosis(self, x: np.ndarray) -> float:
        mean = np.mean(x)
        std  = np.std(x)
        if std == 0:
            return 0.0
        return float(np.mean(((x - mean) / std) ** 4))

    def _print_report(self, results: dict):
        print()
        checks = [
            ("DCT-анализ",    results.get("dct_anomaly",    {})),
            ("LSB-энтропия",  results.get("lsb_entropy",    {})),
            ("Шумовой пол",   results.get("noise_floor",    {})),
            ("Спектр",        results.get("frequency_dist", {})),
            ("Статистика",    results.get("pixel_stats",    {})),
        ]
        for name, r in checks:
            if r.get("available") is False:
                print(f"  {name:<16} — не доступен (нужен scipy)")
            else:
                print(f"  {name:<16} — {r.get('label', '—')}")

        if "diff_analysis" in results:
            print(f"  {'Сравнение':<16} — {results['diff_analysis'].get('label', '—')}")

        # Общий вердикт
        flags = sum([
            results.get("dct_anomaly",  {}).get("anomaly_detected", False),
            results.get("lsb_entropy",  {}).get("suspicious", False),
            results.get("noise_floor",  {}).get("structured_noise", False),
        ])
        print()
        if flags == 0:
            print("  ✅ Признаков watermark не обнаружено (локальный анализ)")
        elif flags == 1:
            print(f"  🟡 Слабые признаки ({flags}/3 индикатора) — вероятно норма")
        else:
            print(f"  🔴 Обнаружены признаки watermark ({flags}/3 индикатора)")
        print(f"{'━'*55}")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".tiff", ".tif",
              ".bmp", ".webp", ".avif", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm",
              ".flv", ".wmv", ".m4v", ".mts", ".m2ts"}
AUDIO_EXTS = {".mp3", ".flac", ".ogg", ".m4a", ".wav", ".aiff", ".wma"}

WM_METHODS = ["ensemble", "gaussian", "dct", "wavelet",
              "photometric", "geometric", "fgsm", "median", "jpeg"]


def main():
    parser = argparse.ArgumentParser(
        description="Удаление метаданных + атаки на invisible watermarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Методы атаки на watermark (--wm-method):
  ensemble    — все методы вместе (рекомендуется)
  gaussian    — гауссовский шум (Stirmark)
  dct         — возмущение DCT-коэффициентов
  wavelet     — шум в вейвлет-домене (DWT)
  photometric — яркость/контраст/гамма
  geometric   — микро-деформации пикселей
  fgsm        — adversarial noise (FGSM-style)
  median      — медианный фильтр
  jpeg        — двойное JPEG-сжатие

Примеры:
  python clean_metadata.py photo.jpg
  python clean_metadata.py photo.jpg --wm-method ensemble --wm-strength 0.5 -c -v
  python clean_metadata.py ./dataset --wm-method ensemble -a -r
  python clean_metadata.py video.mp4 -a
  python clean_metadata.py --check-deps
        """
    )
    parser.add_argument("inputs", nargs="*")
    parser.add_argument("-o", "--output", default="./cleaned")
    parser.add_argument("-a", "--aggressive", action="store_true",
                        help="Агрессивный режим (перекодирование видео, пересоздание пикселей)")
    parser.add_argument("-r", "--randomize-timestamps", action="store_true",
                        help="Рандомизировать timestamps файлов")
    parser.add_argument("-c", "--compare", action="store_true",
                        help="Сравнить метаданные до/после (требует exiftool)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Подробный вывод")
    parser.add_argument("--wm-method", default="ensemble",
                        choices=WM_METHODS, metavar="METHOD",
                        help=f"Метод атаки на watermark [{', '.join(WM_METHODS)}]")
    parser.add_argument("--wm-strength", type=float, default=0.5,
                        metavar="0.0–1.0",
                        help="Сила атаки: 0.3=минимальная, 0.5=баланс, 0.8=максимальная")
    parser.add_argument("--no-watermark-attack", action="store_true",
                        help="Только метаданные, без атаки на watermark")
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument("--analyze", action="store_true",
                        help="Анализировать файлы на признаки watermark (до и после)")

    args = parser.parse_args()

    if args.check_deps or not args.inputs:
        check_deps()
        if not args.inputs:
            parser.print_help()
            return

    if not (0.0 <= args.wm_strength <= 1.0):
        print("[!] --wm-strength должен быть от 0.0 до 1.0")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            for ext_set in [IMAGE_EXTS, VIDEO_EXTS, AUDIO_EXTS]:
                for e in ext_set:
                    files.extend(p.rglob(f"*{e}"))
                    files.extend(p.rglob(f"*{e.upper()}"))
        elif p.is_file():
            files.append(p)
        else:
            print(f"[!] Не найдено: {inp}")

    files = sorted(set(files))
    if not files:
        print("Нет поддерживаемых файлов.")
        return

    print(f"Файлов:        {len(files)}")
    print(f"WM-атака:      {'нет (--no-watermark-attack)' if args.no_watermark_attack else args.wm_method}")
    if not args.no_watermark_attack:
        print(f"WM-strength:   {args.wm_strength}")
    print(f"Вывод:         {output_dir.resolve()}\n")

    analyzer = WatermarkAnalyzer()

    ok_count = 0
    for f in files:
        ext = f.suffix.lower()
        output_path = output_dir / f"{f.stem}_clean{f.suffix}"
        print(f"\n→ {f.name}")

        if args.analyze and ext in IMAGE_EXTS:
            print(f"  [Анализ ДО обработки]")
            analyzer.analyze(f)

        ok = False
        if ext in IMAGE_EXTS:
            if args.no_watermark_attack:
                # Только метаданные
                if has_tool("exiftool"):
                    r, _ = run(["exiftool", "-all=", "-overwrite_original",
                                "-o", str(output_path), str(f)])
                    ok = r
                if not ok:
                    ok = strip_jpeg(f, output_path) if ext in (".jpg", ".jpeg") \
                         else strip_png(f, output_path)
            else:
                ok = process_image(
                    f, output_path,
                    wm_method=args.wm_method,
                    wm_strength=args.wm_strength,
                    aggressive_meta=args.aggressive,
                    verbose=args.verbose
                )
        elif ext in VIDEO_EXTS:
            ok = process_video(f, output_path, args.aggressive)
        elif ext in AUDIO_EXTS:
            ok = process_audio(f, output_path)

        if ok:
            if args.analyze and ext in IMAGE_EXTS and output_path.exists():
                print(f"  [Анализ ПОСЛЕ обработки]")
                analyzer.analyze(output_path, reference_path=f)
            if args.randomize_timestamps:
                randomize_timestamps(output_path)
                print("    [ts] Timestamps рандомизированы")
            if args.compare:
                compare_metadata(f, output_path)
            size_b = f.stat().st_size
            size_a = output_path.stat().st_size
            print(f"  ✓ {size_b/1024:.1f} KB → {size_a/1024:.1f} KB")
            ok_count += 1

    print(f"\n{'═'*55}")
    print(f"✓ Обработано: {ok_count}/{len(files)}")
    print(f"  Результаты: {output_dir.resolve()}")

    if not has_tool("exiftool"):
        print("\n💡 Для полной очистки метаданных установите ExifTool: https://exiftool.org")
    if not HAS_WAVELETS:
        print("💡 Для wavelet-атаки: pip install PyWavelets")
    if not HAS_SCIPY:
        print("💡 Для DCT/geometric атак: pip install scipy")


if __name__ == "__main__":
    main()

