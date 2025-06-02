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

# ---------------------------------------------------------------------------#
#                       BETTER SINGLE-CHAR OCR ENGINE                        #
# ---------------------------------------------------------------------------#
class BestPointsOCR(OCREngine):
    """
    CNN-based recogniser trained on the EMNIST family (white glyph, black
    background, 28×28).  The new pre-processing works with camera frames
    where the glyph may be dark-on-light or light-on-dark and fills most of
    the view.
    """

    def __init__(
        self,
        model_path: str = "Best_points2.h5",
        min_conf: float = 0.60,
        charset: str | List[str] | None = None,
        input_size: Tuple[int, int] = (28, 28),
    ):
        from tensorflow.keras.models import load_model
        self._model = load_model(model_path, compile=False)
        self._min_conf = float(min_conf)
        self._input_size = tuple(map(int, input_size))

        default_charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
        self._charset = list(charset) if charset else list(default_charset)
        if self._model.output_shape[-1] != len(self._charset):
            raise ValueError(
                f"Model expects {self._model.output_shape[-1]} classes, "
                f"but charset has {len(self._charset)}"
            )

    # ------------------------------------------------------------------ #
    # public API                                                          #
    # ------------------------------------------------------------------ #
    def ocr(self, image_bgr: np.ndarray) -> Tuple[str, bool]:
        binary = self._make_emnist_binary(image_bgr)
        crop   = self._largest_blob(binary)
        if crop is None:                      # no glyph found
            return "", False

        canvas = self._centre_on_square(crop)
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)      # soften edges
        canvas = canvas.astype("float32") / 255.0
        canvas = np.expand_dims(canvas, (-1, 0))          # → (1,28,28,1)

        prob = self._model.predict(canvas, verbose=0)[0]
        cls, conf = int(prob.argmax()), float(prob.max())
        print(f"Pred: {self._charset[cls]}   conf={conf:.2f}")

        return (self._charset[cls], True) if conf >= self._min_conf else ("", False)

    # ------------------------------------------------------------------ #
    # helpers                                                             #
    # ------------------------------------------------------------------ #
    def _make_emnist_binary(self, img_bgr: np.ndarray) -> np.ndarray:
        """Return a bin-image with **white glyph on black** (EMNIST style)."""
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # local contrast normalisation → robust to uneven lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        # global Otsu threshold (no inversion yet)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # decide polarity: glyph pixels should be minority
        if np.mean(th) > 127:                 # too many white → invert
            th = cv2.bitwise_not(th)

        # slight closing removes pin-holes inside strokes
        kernel = np.ones((3, 3), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        return th

    def _largest_blob(self, binary: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        # ignore specks under 1 % of the frame
        if w * h < 0.01 * binary.size:
            return None

        pad = int(0.10 * max(w, h))           # 10 % padding
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, binary.shape[1])
        y1 = min(y + h + pad, binary.shape[0])
        return binary[y0:y1, x0:x1]

    def _centre_on_square(self, crop: np.ndarray) -> np.ndarray:
        """Place crop on square black canvas, keep aspect, resize to 28×28."""
        h, w = crop.shape
        side = max(h, w)
        canvas = np.zeros((side, side), dtype=np.uint8)
        y_off = (side - h) // 2
        x_off = (side - w) // 2
        canvas[y_off:y_off + h, x_off:x_off + w] = crop
        return cv2.resize(canvas, self._input_size, interpolation=cv2.INTER_AREA)



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
    ocr_cls: Type[OCREngine] = BestPointsOCR,
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
