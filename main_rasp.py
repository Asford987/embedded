#!/usr/bin/env python3
# live_ocr_cam.py
#
# One-file, modular Picamera2 → OCR → OpenCV overlay loop.
# Swap OCR engines by changing the OCREngine subclass passed to `main()`.

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Type
import os, time, sys

from picamera2 import Picamera2
from libcamera import Transform
import cv2
import numpy as np

# ---------------------------------------------------------------------------#
#                             OCR ABSTRACTION                                #
# ---------------------------------------------------------------------------#
class OCREngine:
    """Abstract base for any OCR backend."""
    def ocr(self, image_bgr: np.ndarray) -> Tuple[str, bool]:
        """
        Run OCR on a BGR image.
        Returns (text, trusted) where *trusted* is a boolean telling the caller
        whether to display the text (e.g. all characters above thresh).
        """
        raise NotImplementedError

# ── paste this over the old TesseractOCR class ──────────────────────────────
class TesseractOCR(OCREngine):
    """
    Tesseract backend that *refuses* to emit text unless
    **all** recognised tokens have confidence ≥ min_conf.
    """
    def __init__(self, min_conf: int = 85, lang: str = "eng"):
        from pytesseract import image_to_data, Output
        self._image_to_data = image_to_data
        self._Output = Output
        self._min_conf = min_conf
        self._lang = lang

    def ocr(self, image_bgr):
        data = self._image_to_data(
            image_bgr,
            lang=self._lang,
            output_type=self._Output.DICT,
            config="--psm 6 --oem 3",
        )

        words: List[str] = []
        for word, conf in zip(data["text"], data["conf"]):
            word = word.strip()
            try:
                conf_val = int(conf)
            except ValueError:
                # Tesseract sometimes returns '' or '-1'
                continue

            # Reject if confidence is too low
            if conf_val < self._min_conf:
                return "", False

            if word:                      # keep non-empty words
                words.append(word)

        if not words:
            return "", False

        return " ".join(words), True
# ────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------#
#                               CAMERA LAYER                                 #
# ---------------------------------------------------------------------------#
@dataclass
class CameraConfig:
    resolution: Tuple[int, int] = (640, 480)
    sharpness: int = 4
    hflip: bool = True
    vflip: bool = True


class Camera:
    """Thin wrapper around Picamera2 for RGB frame capture."""
    def __init__(self, cfg: CameraConfig):
        self._cfg = cfg
        self._picam = Picamera2()
        self._picam.configure(
            self._picam.create_video_configuration(
                main={"size": cfg.resolution, "format": "RGB888"},
                transform=Transform(
                    hflip=cfg.hflip, vflip=cfg.vflip
                ),
            )
        )
        self._picam.set_controls({"Sharpness": cfg.sharpness})
        self._picam.start()
        time.sleep(0.1)

    def get_frame_bgr(self) -> np.ndarray:
        frame_rgb = self._picam.capture_array("main")
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def close(self):
        self._picam.stop()


# ---------------------------------------------------------------------------#
#                                UI LAYER                                    #
# ---------------------------------------------------------------------------#
@dataclass
class UILayout:
    font_scale: float = 1.0
    font_thick: int = 2
    position: Tuple[int, int] = (10, 40)
    color_ok: Tuple[int, int, int] = (0, 255, 0)   # green
    color_bad: Tuple[int, int, int] = (0, 0, 255)  # red
    font: int = cv2.FONT_HERSHEY_SIMPLEX


class OnlineViewer:
    """Shows frames and overlays text when trusted=True."""
    def __init__(self, layout: UILayout):
        self._layout = layout
        self._headless = os.environ.get("DISPLAY") in (None, "")
        if self._headless:
            print("Headless mode detected: no preview will be shown.", file=sys.stderr)

    def show(self, frame_bgr: np.ndarray, text: str, trusted: bool):
        if self._headless:
            return
        if text:
            color = self._layout.color_ok if trusted else self._layout.color_bad
            cv2.putText(
                frame_bgr,
                text,
                self._layout.position,
                self._layout.font,
                self._layout.font_scale,
                color,
                self._layout.font_thick,
                cv2.LINE_AA,
            )
        cv2.imshow("Live OCR", frame_bgr)

    def wait_exit(self, delay_ms: int = 1) -> bool:
        if self._headless:
            time.sleep(delay_ms / 1000)
            return False
        return cv2.waitKey(delay_ms) & 0xFF == ord("q")

    def close(self):
        if not self._headless:
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------#
#                               MAIN LOOP                                    #
# ---------------------------------------------------------------------------#
def main(
    ocr_cls: Type[OCREngine] = TesseractOCR,
    camera_cfg: CameraConfig = CameraConfig(),
    ui_layout: UILayout = UILayout(),
):
    """
    Start the capture → OCR → display loop.
    Pass a different OCREngine subclass to swap OCR model, e.g.:
        main(lambda: EasyOCREngine(threshold=0.8))
    """
    camera = Camera(camera_cfg)
    ocr_engine = ocr_cls()
    viewer = OnlineViewer(ui_layout)

    print("Press ‘q’ in the window (or Ctrl-C) to quit.")
    try:
        while True:
            frame = camera.get_frame_bgr()
            text, trusted = ocr_engine.ocr(frame)
            viewer.show(frame, text, trusted)

            if viewer.wait_exit():
                break
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        camera.close()


# ---------------------------------------------------------------------------#
# Launch if executed directly
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
