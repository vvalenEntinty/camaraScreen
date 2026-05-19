#!/usr/bin/env python3
"""
Controlador de mouse con gestos de mano.
Instalacion: pip install -r requirements.txt
"""

import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import urllib.request
import os
import math
import ctypes
import threading
import subprocess
import queue
from collections import deque
import argparse
from typing import Optional, Tuple


# Movimiento de mouse directo via Win32 API (sin overhead de pyautogui)
def _move_cursor(x: int, y: int):
    ctypes.windll.user32.SetCursorPos(x, y)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


def _get_monitor_hz() -> int:
    """Refresh rate del monitor primario via Win32 GetDeviceCaps (VREFRESH=116)."""
    hdc = ctypes.windll.user32.GetDC(0)
    hz  = ctypes.windll.gdi32.GetDeviceCaps(hdc, 116)
    ctypes.windll.user32.ReleaseDC(0, hdc)
    return max(hz, 60)


class _MouseMover:
    """Mueve el cursor en un hilo dedicado a la frecuencia del monitor."""

    def __init__(self, hz: int):
        self._interval = 1.0 / hz
        # half-life 8 ms → ~95 % de convergencia dentro de un frame de 30 fps (~33 ms)
        self._alpha  = 1.0 - 0.5 ** (self._interval / 0.008)
        self._lock   = threading.Lock()
        self._tx     = 0.0
        self._ty     = 0.0
        self._active = False
        self._stop   = threading.Event()
        # Resolución de timer a 1 ms para que time.sleep sea preciso a alta frecuencia
        ctypes.windll.winmm.timeBeginPeriod(1)
        threading.Thread(target=self._run, daemon=True).start()
        print(f"  Mouse mover: {hz} Hz  (alpha={self._alpha:.3f})")

    def set_target(self, x: int, y: int):
        with self._lock:
            self._tx = float(x)
            self._ty = float(y)
            self._active = True

    def set_inactive(self):
        with self._lock:
            self._active = False

    def stop(self):
        self._stop.set()
        ctypes.windll.winmm.timeEndPeriod(1)

    def _run(self):
        cx = cy = 0.0
        initialized = False
        while not self._stop.is_set():
            t0 = time.perf_counter()
            with self._lock:
                active = self._active
                tx, ty = self._tx, self._ty
            if active:
                if not initialized:
                    cx, cy = tx, ty
                    initialized = True
                cx += self._alpha * (tx - cx)
                cy += self._alpha * (ty - cy)
                _move_cursor(int(cx), int(cy))
            else:
                initialized = False
            rem = self._interval - (time.perf_counter() - t0)
            if rem > 0.0001:
                time.sleep(rem)

# ─────────────────────────── Auto-encuadre ────────────────────────────

class AutoFramer:
    FACE_SCALE   = 4.5
    CROP_MIN     = 0.50
    CROP_MAX     = 0.98
    SIZE_LERP    = 0.12
    POS_LERP      = 0.10
    POS_SOFT_ZONE = 0.04   # banda suave: lerp escala de 0 a 1 dentro de este rango

    def __init__(self, fw: int, fh: int):
        self._fw = fw
        self._fh = fh
        self._cw = int(fw * 0.75)
        self._ch = int(fh * 0.75)
        self._cx = float(fw // 2)
        self._cy = float(fh // 2)

    def update(self, face_nx: float, face_ny: float, face_pw: float = 0.0):
        if face_pw > 0:
            target_cw = int(self._fw * face_pw * self.FACE_SCALE)
            target_cw = max(int(self._fw * self.CROP_MIN),
                            min(int(self._fw * self.CROP_MAX), target_cw))
            target_ch = int(target_cw * self._fh / self._fw)
            self._cw  = int(self._cw + self.SIZE_LERP * (target_cw - self._cw))
            self._ch  = int(self._ch + self.SIZE_LERP * (target_ch - self._ch))
        dx   = face_nx * self._fw - self._cx
        dy   = face_ny * self._fh - self._cy
        dist = math.sqrt(dx * dx + dy * dy)
        t    = min(1.0, max(0.0, dist / (self._fw * self.POS_SOFT_ZONE) - 1.0))
        self._cx += self.POS_LERP * t * dx
        self._cy += self.POS_LERP * t * dy

    def get_crop(self) -> Tuple[int, int, int, int]:
        x0 = max(0, int(self._cx) - self._cw // 2)
        y0 = max(0, int(self._cy) - self._ch // 2)
        x1 = min(self._fw, x0 + self._cw)
        y1 = min(self._fh, y0 + self._ch)
        if x1 - x0 < self._cw:
            x0 = max(0, x1 - self._cw)
        if y1 - y0 < self._ch:
            y0 = max(0, y1 - self._ch)
        return x0, y0, x1, y1


# ═══════════════════════════ CONFIGURACIÓN ════════════════════════════
CAMERA_ID   = 0
RESOLUTIONS = [(960, 540), (640, 360), (640, 480)]

SCREEN_W, SCREEN_H = pyautogui.size()

# One Euro Filter — reduce jitter sin agregar lag
OEF_MINCUTOFF = 2.5    # menor = más suavizado cuando la mano está quieta
OEF_BETA      = 2.5    # mayor = más responsivo cuando la mano se mueve rápido
MARGIN              = 0.12
FIST_MAX_OPEN       = 1
SCROLL_SPEED_FACTOR = 3000  # scrolls/segundo por unidad^2 de offset (escala cuadratica)
SCROLL_DEADZONE     = 0.04  # zona muerta central antes de empezar a scrollear
PINCH_THRESHOLD     = 0.06  # distancia normalizada pulgar-indice para detectar pinch
DRAG_PINCH_HOLD     = 0.5   # segundos manteniendo pinch para activar arrastre
ZOOM_SENSITIVITY    = 50    # clicks de zoom por unidad de distancia entre manos
FACE_DET_SCALE      = 0.5  # fracción del frame para detección de cara (menos pixels = más rápido)
DISPLAY_SCALE       = 1.5  # escala del preview (1.0 = tamaño nativo de la cámara)
HUD_BAR_H           = 90   # altura en px de la franja de info debajo del video
# ══════════════════════════════════════════════════════════════════════

MODEL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
SOUND_SEXY  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sonidos", "anime.m4a")
SOUND_PERON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sonidos", "peron.m4a")
SOUND_COOLDOWN = 3.0   # segundos entre reproducciones
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# ──────────────────────────── Audio Peron ─────────────────────────────
_peron_proc: Optional[subprocess.Popen] = None

def start_peron_music():
    global _peron_proc
    stop_peron_music()
    uri = "file:///" + os.path.abspath(SOUND_PERON).replace("\\", "/")
    script = (
        "Add-Type -AssemblyName presentationCore;"
        f"$p=New-Object System.Windows.Media.MediaPlayer;"
        f"$p.Open([uri]'{uri}');"
        "$p.MediaEnded+={{$p.Position=[TimeSpan]::Zero;$p.Play()}};"
        "$p.Play();"
        "while($true){{Start-Sleep 1}}"
    )
    _peron_proc = subprocess.Popen(
        ["powershell", "-WindowStyle", "Hidden", "-NonInteractive", "-Command", script],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

def stop_peron_music():
    global _peron_proc
    if _peron_proc and _peron_proc.poll() is None:
        _peron_proc.terminate()
    _peron_proc = None

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

# ──────────────────────────── Cámara ──────────────────────────────────

class CameraCapture:
    """Lee frames en un hilo dedicado para que siempre tengamos el más fresco."""
    def __init__(self, camera_id: int, resolutions: list):
        self._cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for w, h in resolutions:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            got_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            got_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if abs(got_w - w) <= 4 and abs(got_h - h) <= 4:
                self.width  = got_w
                self.height = got_h
                print(f"  Cámara resolución: {self.width}x{self.height}")
                break
        else:
            self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  Cámara resolución (fallback): {self.width}x{self.height}")
        for target_fps in (30, 25, 24):
            self._cap.set(cv2.CAP_PROP_FPS, target_fps)
            got = self._cap.get(cv2.CAP_PROP_FPS)
            if abs(got - target_fps) <= 2:
                print(f"  Cámara FPS: {got:.0f}")
                break
        self._frame: Optional[np.ndarray] = None
        self._frame_id: int = 0
        self._lock  = threading.Lock()
        self._alive = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._alive:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        with self._lock:
            if self._frame is None:
                return False, None, -1
            return True, self._frame.copy(), self._frame_id

    @property
    def opened(self) -> bool:
        return self._cap.isOpened()

    def release(self):
        self._alive = False
        self._thread.join(timeout=1.0)
        self._cap.release()


_TIPS      = [8, 12, 16, 20]
_PIPS      = [6, 10, 14, 18]
_PALM_PTS  = [0, 5, 9, 13, 17]
_FINGER_NAMES = ["Pulgar", "Indice", "Medio", "Anular", "Menique"]


# ──────────────────────────── Modelo ──────────────────────────────────

def _report_progress(count, block_size, total_size):
    pct = int(count * block_size * 100 / total_size)
    print(f"\r  Descargando... {min(pct, 100)}%", end="", flush=True)


def play_sound(path: str):
    uri = "file:///" + os.path.abspath(path).replace("\\", "/")
    script = (
        "Add-Type -AssemblyName presentationCore;"
        f"$p=New-Object System.Windows.Media.MediaPlayer;"
        f"$p.Open([uri]'{uri}');"
        "$p.Play();"
        "Start-Sleep 10"
    )
    subprocess.Popen(
        ["powershell", "-WindowStyle", "Hidden", "-NonInteractive", "-Command", script],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def is_pointing(lm, hand_label: str) -> bool:
    """👉 indice extendido, resto cerrado."""
    fingers = fingers_open(lm, hand_label)
    return fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]


def is_ok_sign(lm, hand_label: str) -> bool:
    """👌 pinch pulgar+indice + medio/anular/menique extendidos."""
    fingers = fingers_open(lm, hand_label)
    return pinch_distance(lm) < PINCH_THRESHOLD and fingers[2] and fingers[3] and fingers[4]


def hands_united(lm_point, lm_ok) -> bool:
    """True cuando la punta del índice (👉) está dentro del círculo del OK (👌)."""
    tip = lm_point[8]
    ok_cx = (lm_ok[4].x + lm_ok[8].x) / 2
    ok_cy = (lm_ok[4].y + lm_ok[8].y) / 2
    dist = math.sqrt((tip.x - ok_cx) ** 2 + (tip.y - ok_cy) ** 2)
    return dist < 0.08


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Modelo no encontrado. Descargando (~14 MB)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, _report_progress)
            print("\nModelo descargado OK.")
        except Exception as e:
            print(f"\nERROR descargando modelo: {e}")
            raise


# ──────────────────────────── Helpers ─────────────────────────────────

def fingers_open(lm, hand_label: str) -> list:
    result = [lm[tip].y < lm[pip].y for tip, pip in zip(_TIPS, _PIPS)]
    thumb = lm[4].x < lm[3].x if hand_label == "Right" else lm[4].x > lm[3].x
    return [thumb] + result


def pinch_distance(lm) -> float:
    dx = lm[4].x - lm[8].x
    dy = lm[4].y - lm[8].y
    return math.sqrt(dx * dx + dy * dy)


def palm_pos(lm) -> Tuple[float, float]:
    return (
        float(np.mean([lm[i].x for i in _PALM_PTS])),
        float(np.mean([lm[i].y for i in _PALM_PTS])),
    )


def to_screen(nx: float, ny: float) -> Tuple[int, int]:
    nx = float(np.clip((nx - MARGIN) / (1 - 2 * MARGIN), 0, 1))
    ny = float(np.clip((ny - MARGIN) / (1 - 2 * MARGIN), 0, 1))
    return int(nx * SCREEN_W), int(ny * SCREEN_H)


def draw_hand(frame, landmarks, is_pinch: bool = False):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (80, 230, 120), 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 4, (50, 150, 20), -1)
    if is_pinch:
        # línea entre pulgar e índice en color pinch
        cv2.line(frame, pts[4], pts[8], (0, 200, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, pts[4], 8, (0, 200, 255), -1)
        cv2.circle(frame, pts[8], 8, (0, 200, 255), -1)


class _LowPassFilter:
    def __init__(self):
        self._y: Optional[float] = None

    def filter(self, x: float, alpha: float) -> float:
        if self._y is None:
            self._y = x
        self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y

    def reset(self):
        self._y = None


class OneEuroFilter:
    """Filtra jitter sin agregar lag. Adapta el suavizado según la velocidad."""
    def __init__(self, mincutoff: float = 1.0, beta: float = 0.007, dcutoff: float = 1.0):
        self.mincutoff = mincutoff
        self.beta      = beta
        self.dcutoff   = dcutoff
        self._x  = _LowPassFilter()
        self._dx = _LowPassFilter()
        self._last_t: Optional[float] = None
        self._last_x: Optional[float] = None

    def _alpha(self, cutoff: float, freq: float) -> float:
        te  = 1.0 / freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, t: float) -> float:
        freq = 1.0 / (t - self._last_t) if self._last_t and t > self._last_t else 30.0
        self._last_t = t

        dx = (x - self._last_x) * freq if self._last_x is not None else 0.0
        self._last_x = x

        edx     = self._dx.filter(dx, self._alpha(self.dcutoff, freq))
        cutoff  = self.mincutoff + self.beta * abs(edx)
        return self._x.filter(x, self._alpha(cutoff, freq))

    def reset(self):
        self._x.reset()
        self._dx.reset()
        self._last_t = None
        self._last_x = None


class Smoother:
    """Par de One Euro Filters para X e Y."""
    def __init__(self):
        self._fx = OneEuroFilter(mincutoff=OEF_MINCUTOFF, beta=OEF_BETA)
        self._fy = OneEuroFilter(mincutoff=OEF_MINCUTOFF, beta=OEF_BETA)

    def update(self, x: int, y: int, t: float) -> Tuple[int, int]:
        return int(self._fx.filter(x, t)), int(self._fy.filter(y, t))

    def clear(self):
        self._fx.reset()
        self._fy.reset()


# ──────────────────────── Controlador principal ────────────────────────

class HandController:
    def __init__(self, mover: '_MouseMover'):
        self._mover  = mover
        self._smooth = Smoother()

        self._fist_active   = False
        self._last_scroll_t = 0.0
        self._scroll_accum  = 0.0
        self._hscroll_accum = 0.0

        self._pinch_active = False
        self._pinch_start  = 0.0
        self._dragging     = False
        self._peace_active = False

        self._zoom_active   = False
        self._zoom_ref_dist = 0.0
        self._zoom_accum    = 0.0
        self._pinch_blocked = False   # bloqueado tras salir de zoom hasta soltar

        self.status         = "Iniciando..."
        self._gesture_text  = ""
        self._gesture_until = 0.0

    def process(self, lm, hand_label: str, now: float):
        fingers = fingers_open(lm, hand_label)
        n_open  = sum(fingers)

        is_fist  = n_open <= FIST_MAX_OPEN
        is_peace = (fingers[1] and fingers[2]
                    and not fingers[3] and not fingers[4]
                    and not is_fist)
        is_pinch = (pinch_distance(lm) < PINCH_THRESHOLD
                    and not is_fist and not is_peace)

        px, py = palm_pos(lm)
        sx, sy = to_screen(px, py)
        mx, my = self._smooth.update(sx, sy, now)

        if is_fist:
            self._handle_fist_scroll(px, py, now)
            if self._dragging:
                pyautogui.mouseUp()
                self._dragging = False
            self._pinch_active = False
            self._handle_peace(False)
            self.status = "Scroll (puno)"
        elif is_peace:
            self._handle_peace(True)
            if self._fist_active:
                self._fist_active   = False
                self._scroll_accum  = 0.0
                self._hscroll_accum = 0.0
            self.status = "Peron mode ✌"
        else:
            self._handle_peace(False)
            if self._fist_active:
                self._fist_active   = False
                self._scroll_accum  = 0.0
                self._hscroll_accum = 0.0
            self._handle_pinch(is_pinch, mx, my, now)
            self._move_mouse(is_pinch, mx, my, now, n_open)

        return mx, my, fingers, is_fist, is_pinch, is_peace

    def no_hand(self, now: float):
        self._fist_active   = False
        self._scroll_accum  = 0.0
        self._hscroll_accum = 0.0
        self._pinch_active  = False
        if self._dragging:
            pyautogui.mouseUp()
            self._dragging = False
        self._zoom_active   = False
        self._zoom_accum    = 0.0
        self._pinch_blocked = False
        self._handle_peace(False)
        self._smooth.clear()
        self._mover.set_inactive()
        self.status = "Sin mano detectada"

    def cleanup(self):
        try:
            if self._dragging:
                pyautogui.mouseUp()
            if self._zoom_active:
                pyautogui.keyUp('ctrl')
        except pyautogui.FailSafeException:
            pass
        self._handle_peace(False)

    def _flash(self, text: str, dur: float = 0.9):
        self._gesture_text  = text
        self._gesture_until = time.time() + dur

    def _handle_peace(self, active: bool):
        if active and not self._peace_active:
            self._peace_active = True
            start_peron_music()
        elif not active and self._peace_active:
            self._peace_active = False
            stop_peron_music()

    def _handle_pinch(self, is_pinch: bool, mx: int, my: int, now: float):
        if self._zoom_active:
            self._pinch_active = False
            return
        if self._pinch_blocked:
            if not is_pinch:
                self._pinch_blocked = False  # mano suelta → desbloquear
            return
        if is_pinch and not self._pinch_active:
            self._pinch_active = True
            self._pinch_start  = now
        elif is_pinch and self._pinch_active:
            if not self._dragging and now - self._pinch_start >= DRAG_PINCH_HOLD:
                pyautogui.mouseDown()
                self._dragging = True
                self._flash("Arrastrando...", dur=9999)
        elif not is_pinch and self._pinch_active:
            self._pinch_active = False
            if self._dragging:
                pyautogui.mouseUp()
                self._dragging = False
                self._flash("Soltar arrastre")
            else:
                pyautogui.click(mx, my)
                self._flash("Click Izquierdo")

    def _handle_fist_scroll(self, palm_x: float, palm_y: float, now: float):
        if not self._fist_active:
            self._fist_active   = True
            self._scroll_accum  = 0.0
            self._hscroll_accum = 0.0
            self._last_scroll_t = now
            return

        dt = now - self._last_scroll_t
        self._last_scroll_t = now

        def _axis_speed(offset):
            m = abs(offset)
            if m <= SCROLL_DEADZONE:
                return 0.0
            e = m - SCROLL_DEADZONE
            return math.copysign(e * e * SCROLL_SPEED_FACTOR, offset)

        self._scroll_accum  += _axis_speed(palm_y - 0.5) * dt
        self._hscroll_accum += _axis_speed(palm_x - 0.5) * dt

        clicks_v = int(self._scroll_accum)
        if clicks_v:
            pyautogui.scroll(-clicks_v)
            self._scroll_accum -= clicks_v

        clicks_h = int(self._hscroll_accum)
        if clicks_h:
            pyautogui.hscroll(clicks_h)
            self._hscroll_accum -= clicks_h

    def _move_mouse(self, is_pinch: bool, mx: int, my: int, now: float, n_open: int):
        if self._dragging:
            self._mover.set_target(mx, my)
            self.status = f"Arrastrando ({mx}, {my})"
        elif is_pinch:
            held = now - self._pinch_start
            self._mover.set_target(mx, my)
            self.status = f"Pinch {held:.1f}s / {DRAG_PINCH_HOLD:.0f}s"
        else:
            self._mover.set_target(mx, my)
            self.status = f"({mx}, {my})  dedos: {n_open}/5"

    def update_zoom(self, active: bool, dist: float, now: float):
        if active:
            self._pinch_active = False   # cancelar click pendiente
            if not self._zoom_active:
                self._zoom_active   = True
                self._zoom_ref_dist = dist
                self._zoom_accum    = 0.0
            else:
                delta = dist - self._zoom_ref_dist
                self._zoom_ref_dist = dist
                self._zoom_accum += delta * ZOOM_SENSITIVITY
                clicks = int(self._zoom_accum)
                if clicks:
                    pyautogui.keyDown('ctrl')
                    pyautogui.scroll(clicks)
                    pyautogui.keyUp('ctrl')
                    self._zoom_accum -= clicks
        else:
            if self._zoom_active:
                self._zoom_active   = False
                self._zoom_accum    = 0.0
                self._pinch_active  = False
                self._pinch_blocked = True   # requiere soltar y re-pinchar

    @property
    def gesture_overlay(self) -> Optional[str]:
        return self._gesture_text if time.time() < self._gesture_until else None

    @property
    def dragging(self) -> bool:
        return self._dragging

    @property
    def zooming(self) -> bool:
        return self._zoom_active


# ──────────────────────────── Overlay ────────────────────────────────

class HandOverlay:
    """Ventana translucida siempre encima que muestra la posicion de la mano."""
    SIZE = 180

    def __init__(self):
        self._q      = queue.Queue(maxsize=2)
        self._alive  = True
        self._paused = False
        self._done   = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def update(self, nx: float, ny: float, visible: bool,
               nx2: float = 0.5, ny2: float = 0.5, visible2: bool = False):
        try:
            self._q.put_nowait((nx, ny, visible, nx2, ny2, visible2))
        except queue.Full:
            pass

    def set_paused(self, paused: bool):
        self._paused = paused

    def close(self):
        self._alive = False
        self._done.wait(timeout=2.0)

    def _run(self):
        import tkinter as tk

        root = tk.Tk()
        root.overrideredirect(True)
        root.wm_attributes("-topmost", True)
        root.wm_attributes("-alpha", 0.85)

        S   = self.SIZE
        gap = 20
        sh  = root.winfo_screenheight()
        root.geometry(f"{S}x{S}+{gap}+{sh - S - gap}")

        # Invisible para software de captura/grabacion (igual que RivaTuner)
        root.update_idletasks()
        hwnd = root.winfo_id()
        ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x00000011)

        cv = tk.Canvas(root, width=S, height=S, bg="#0a0a1a", highlightthickness=0)
        cv.pack()

        M    = int(S * MARGIN)   # zona de margen (fuera del area activa del mouse)
        EDGE = "#00e5cc"         # borde exterior
        ZONE = "#1a4040"         # rectangulo zona activa

        nx_v   = [0.5]
        ny_v   = [0.5]
        vis_v  = [False]
        nx2_v  = [0.5]
        ny2_v  = [0.5]
        vis2_v = [False]

        EDGE2 = "#ff8c00"  # naranja para la segunda mano

        def redraw():
            if not self._alive:
                root.destroy()
                self._done.set()
                return
            try:
                while True:
                    nx_v[0], ny_v[0], vis_v[0], nx2_v[0], ny2_v[0], vis2_v[0] = self._q.get_nowait()
            except queue.Empty:
                pass

            cv.delete("all")

            paused = self._paused
            border = "#ff4444" if paused else EDGE

            # fondo
            cv.create_rectangle(0, 0, S, S, fill="#0a0a1a", outline="")
            # zona activa del mouse (sin margen)
            cv.create_rectangle(M, M, S - M, S - M, fill="", outline=ZONE, width=1)
            # borde exterior con esquinas marcadas
            cv.create_rectangle(1, 1, S - 1, S - 1, outline=border, width=2)
            # esquinas reforzadas
            c = 10
            for x0, y0, x1, y1 in [(1,1,c,1),(1,1,1,c),(S-c,1,S-1,1),(S-1,1,S-1,c),
                                     (1,S-c,1,S-1),(1,S-1,c,S-1),(S-c,S-1,S-1,S-1),(S-1,S-c,S-1,S-1)]:
                cv.create_line(x0, y0, x1, y1, fill=border, width=3)

            if paused:
                cv.create_text(S // 2, S // 2, text="PAUSA",
                               fill="#ff4444", font=("Arial", 13, "bold"))

            if vis_v[0] and not paused:
                dx = int(nx_v[0] * S)
                dy = int(ny_v[0] * S)
                r  = 7
                cv.create_oval(dx-r+1, dy-r+1, dx+r+1, dy+r+1, fill="#003333", outline="")
                cv.create_oval(dx-r, dy-r, dx+r, dy+r, fill=EDGE, outline="")

            if vis2_v[0] and not paused:
                dx2 = int(nx2_v[0] * S)
                dy2 = int(ny2_v[0] * S)
                r   = 7
                cv.create_oval(dx2-r+1, dy2-r+1, dx2+r+1, dy2+r+1, fill="#331a00", outline="")
                cv.create_oval(dx2-r, dy2-r, dx2+r, dy2+r, fill=EDGE2, outline="")

            root.after(30, redraw)

        redraw()
        root.mainloop()


# ──────────────────────────── HUD ─────────────────────────────────────

def _choose_face(faces, preferred_nxy, cap_w: int, cap_h: int):
    """Devuelve la cara más cercana a preferred_nxy (coordenadas norm. 0-1), o faces[0]."""
    if preferred_nxy is None or len(faces) <= 1:
        return faces[0]
    inv    = 1.0 / FACE_DET_SCALE
    pref_x = preferred_nxy[0] * cap_w
    pref_y = preferred_nxy[1] * cap_h
    best, best_d = faces[0], float('inf')
    for f in faces:
        fx, fy, fw, fh = f
        cx = (fx + fw / 2) * inv
        cy = (fy + fh / 2) * inv
        d  = (cx - pref_x) ** 2 + (cy - pref_y) ** 2
        if d < best_d:
            best, best_d = f, d
    return best


_HUD_LEGENDS = [
    "Mano abierta -> mover",
    "Pinch        -> click izq",
    "Pinch 0.5s   -> arrastrar",
    "Puno         -> scroll",
    "2x Pinch     -> zoom",
    "V (paz)      -> Peron",
]

def draw_hud(frame, ctrl: HandController, fingers, is_fist, is_pinch, is_peace, now):
    h, w = frame.shape[:2]

    for i, name in enumerate(_FINGER_NAMES):
        if i < len(fingers):
            color = (80, 210, 80) if fingers[i] else (60, 60, 200)
            icon  = "+" if fingers[i] else "-"
            cv2.putText(frame, f"{icon} {name}", (10, 28 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    if ctrl.zooming:
        ind_color, ind_label = (255, 140, 0), "ZOOM"
    elif ctrl.dragging:
        ind_color, ind_label = (0, 200, 200), "DRAG"
    elif is_fist:
        ind_color, ind_label = (200, 160, 0), "SCROLL"
    elif is_peace:
        ind_color, ind_label = (0, 120, 255), "PERON"
    elif is_pinch:
        ind_color, ind_label = (0, 200, 255), "PINCH"
    else:
        ind_color, ind_label = (80, 200, 80), "ABIERTA"

    cv2.circle(frame, (w - 25, 25), 14, ind_color, -1)
    cv2.putText(frame, ind_label, (w - 80, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, ind_color, 1, cv2.LINE_AA)

    gesture = ctrl.gesture_overlay
    if gesture:
        tw = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)[0][0]
        alpha = min(1.0, (ctrl._gesture_until - now) / 0.25)
        col   = (int(255 * alpha), int(255 * alpha), 50)
        cv2.putText(frame, gesture, ((w - tw) // 2, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, col, 2, cv2.LINE_AA)


def _enumerate_cameras(max_check: int = 5) -> list:
    """Prueba índices 0..max_check-1 y devuelve los que se abren correctamente."""
    found = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found if found else [0]


# ──────────────────────────── Tray icon (pystray) ─────────────────────

class _TrayIcon:
    """Icono en bandeja del sistema via pystray."""

    def __init__(self, stop_event: threading.Event, camera_ids: list, initial_cam: int):
        self._stop   = stop_event
        self._icon   = None
        self._paused    = False
        self._autoframe = False
        self._camera_ids    = camera_ids
        self._current_cam   = initial_cam
        self._requested_cam = initial_cam
        self._lock      = threading.Lock()

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def autoframe(self) -> bool:
        with self._lock:
            return self._autoframe

    @property
    def requested_camera(self) -> int:
        with self._lock:
            return self._requested_cam

    def acknowledge_camera(self, cam_id: int):
        with self._lock:
            self._current_cam = cam_id

    def run(self):
        import pystray
        from PIL import Image, ImageDraw

        img  = Image.new("RGBA", (64, 64), (10, 10, 26, 255))
        draw = ImageDraw.Draw(img)
        draw.ellipse([14, 14, 50, 50], fill=(0, 229, 204, 255))

        def on_toggle(icon, item):
            with self._lock:
                self._paused = not self._paused

        def on_autoframe(icon, item):
            with self._lock:
                self._autoframe = not self._autoframe

        def on_quit(icon, item):
            icon.stop()
            self._stop.set()

        def make_cam_handler(cam_id):
            def handler(icon, item):
                with self._lock:
                    self._requested_cam = cam_id
            return handler

        def make_cam_checked(cam_id):
            def checked(item):
                with self._lock:
                    return self._current_cam == cam_id
            return checked

        cam_items = [
            pystray.MenuItem(
                f"Cámara {cam_id}",
                make_cam_handler(cam_id),
                checked=make_cam_checked(cam_id),
                radio=True,
            )
            for cam_id in self._camera_ids
        ]

        menu = pystray.Menu(
            pystray.MenuItem(
                lambda item: "Reanudar" if self._paused else "Pausar",
                on_toggle,
            ),
            pystray.MenuItem(
                lambda item: "Auto-encuadre: ON" if self._autoframe else "Auto-encuadre: OFF",
                on_autoframe,
            ),
            pystray.MenuItem("Cámara", pystray.Menu(*cam_items)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Cerrar Hand Controller", on_quit),
        )

        self._icon = pystray.Icon("HandController", img, "Hand Controller", menu)
        self._icon.run()

    def stop(self):
        if self._icon:
            self._icon.stop()


# ─────────────────────────────── Main ─────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true",
                        help="Sin preview de camara; icono en bandeja del sistema")
    args     = parser.parse_args()
    headless = args.headless

    ensure_model()

    last_sound_t = 0.0
    stop_flag    = threading.Event()

    print("Buscando cámaras disponibles...")
    available_cams = _enumerate_cameras()
    print(f"  Cámaras encontradas: {available_cams}")
    current_cam_id = CAMERA_ID if CAMERA_ID in available_cams else available_cams[0]
    cap = CameraCapture(current_cam_id, RESOLUTIONS)

    if not cap.opened:
        print(f"ERROR: No se pudo abrir la camara {current_cam_id}")
        return

    hz    = _get_monitor_hz()
    mover = _MouseMover(hz)
    ctrl  = HandController(mover)
    overlay = HandOverlay()

    tray = _TrayIcon(stop_flag, available_cams, current_cam_id)
    threading.Thread(target=tray.run, daemon=True).start()

    _ocl = cv2.ocl.haveOpenCL()
    if _ocl:
        cv2.ocl.setUseOpenCL(True)
        print(f"  OpenCL: habilitado  [{cv2.ocl.Device.getDefault().name()}]")
    else:
        print("  OpenCL: no disponible, procesando en CPU")

    BaseOptions         = mp.tasks.BaseOptions
    HandLandmarker      = mp.tasks.vision.HandLandmarker
    HandLandmarkerOpts  = mp.tasks.vision.HandLandmarkerOptions
    RunningMode         = mp.tasks.vision.RunningMode

    # Resultado más reciente de MediaPipe (actualizado desde el hilo de callback)
    _latest = [None]
    _lock   = threading.Lock()

    def on_result(result, _img, _ts):
        with _lock:
            _latest[0] = result

    options = HandLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=on_result,
        num_hands=2,
        min_hand_detection_confidence=0.70,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.60,
    )

    print("=" * 52)
    print("  Controlador de Mouse por Gestos  |  ESC = salir")
    print("=" * 52)
    print("  Mano abierta        -> mover mouse")
    print("  Pinch pulgar+indice -> click izquierdo")
    print("  Puno cerrado        -> scroll vertical (mover mano)")
    print("  V (paz) ✌           -> musica de Peron")
    print("  Mouse esquina sup-izq -> emergencia (failsafe)")
    print("=" * 52)

    t0 = time.time()
    last_frame_id  = -1
    autoframe_on   = False
    autoframer: Optional[AutoFramer] = None
    _face_det      = None
    preferred_face: Optional[Tuple[float, float]] = None
    last_faces     = []
    last_select_t  = 0.0
    select_hold_t  = 0.0  # momento en que empezó el gesto de selección
    _face_pos_buf: deque = deque(maxlen=7)

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.opened:
            # ── Cambio de cámara ──────────────────────────────────────
            req_cam = tray.requested_camera
            if req_cam != current_cam_id:
                cap.release()
                new_cap = CameraCapture(req_cam, RESOLUTIONS)
                if new_cap.opened:
                    cap            = new_cap
                    current_cam_id = req_cam
                    tray.acknowledge_camera(req_cam)
                    if autoframe_on and autoframer is not None:
                        autoframer = AutoFramer(cap.width, cap.height)
                        _face_pos_buf.clear()
                else:
                    new_cap.release()
                    cap = CameraCapture(current_cam_id, RESOLUTIONS)
                    tray.acknowledge_camera(current_cam_id)
                last_frame_id = -1

            ret, frame, frame_id = cap.read()
            if not ret:
                time.sleep(0.001)
                continue

            frame = cv2.flip(frame, 1)
            now   = time.time()

            # ── Auto-encuadre ──────────────────────────────────────────
            autoframe_on = tray.autoframe
            if autoframe_on and _face_det is None:
                _face_det  = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                autoframer = AutoFramer(cap.width, cap.height)
            elif not autoframe_on and _face_det is not None:
                _face_det  = None
                autoframer = None
                _face_pos_buf.clear()

            if autoframe_on and autoframer is not None:
                fd_w  = int(cap.width  * FACE_DET_SCALE)
                fd_h  = int(cap.height * FACE_DET_SCALE)
                small = cv2.resize(frame, (fd_w, fd_h))
                gray  = cv2.cvtColor(cv2.UMat(small) if _ocl else small,
                                     cv2.COLOR_BGR2GRAY)
                min_s = max(15, int(60 * FACE_DET_SCALE))
                faces = _face_det.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_s, min_s)
                )
                if len(faces) > 0:
                    last_faces = faces
                    chosen     = _choose_face(last_faces, preferred_face, cap.width, cap.height)
                    fx, fy, fw, fh = chosen
                    inv = 1.0 / FACE_DET_SCALE
                    _face_pos_buf.append(((fx + fw / 2) * inv / cap.width,
                                          (fy + fh / 2) * inv / cap.height))
                    avg_nx = sum(p[0] for p in _face_pos_buf) / len(_face_pos_buf)
                    avg_ny = sum(p[1] for p in _face_pos_buf) / len(_face_pos_buf)
                    autoframer.update(avg_nx, avg_ny, fw * inv / cap.width)
                x0, y0, x1, y1 = autoframer.get_crop()
                frame_mp = frame[y0:y1, x0:x1]
            else:
                frame_mp = frame

            # ── Enviar a MediaPipe ─────────────────────────────────────
            if frame_id != last_frame_id:
                last_frame_id = frame_id
                if autoframe_on:
                    src    = cv2.UMat(frame_mp) if _ocl else frame_mp
                    mp_src = cv2.resize(src, (cap.width, cap.height))
                else:
                    mp_src = cv2.UMat(frame_mp) if _ocl else frame_mp
                rgb = cv2.cvtColor(mp_src, cv2.COLOR_BGR2RGB)
                if _ocl:
                    rgb = rgb.get()
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms    = int((now - t0) * 1000)
                landmarker.detect_async(mp_image, ts_ms)

            with _lock:
                result = _latest[0]

            paused = tray.paused if tray else False
            overlay.set_paused(paused)

            try:
                if paused:
                    ctrl.no_hand(now)
                    overlay.update(0.5, 0.5, False)
                elif result and result.hand_landmarks and result.handedness:
                    lm         = result.hand_landmarks[0]
                    raw_label  = result.handedness[0][0].category_name
                    hand_label = "Left" if raw_label == "Right" else "Right"

                    mx, my, fingers, is_fist, is_pinch, is_peace = ctrl.process(lm, hand_label, now)
                    px, py = palm_pos(lm)
                    if not headless:
                        draw_hand(frame_mp, lm, is_pinch)
                        draw_hud(frame_mp if autoframe_on else frame, ctrl, fingers, is_fist, is_pinch, is_peace, now)

                    if len(result.hand_landmarks) >= 2:
                        lm1    = result.hand_landmarks[1]
                        raw1   = result.handedness[1][0].category_name
                        label1 = "Left" if raw1 == "Right" else "Right"
                        px1, py1 = palm_pos(lm1)
                        overlay.update(px, py, True, px1, py1, True)
                        if not headless:
                            draw_hand(frame_mp, lm1)

                        both_pinching = (pinch_distance(lm) < PINCH_THRESHOLD and
                                         pinch_distance(lm1) < PINCH_THRESHOLD)
                        if both_pinching:
                            cx0  = (lm[4].x  + lm[8].x)  / 2
                            cy0  = (lm[4].y  + lm[8].y)  / 2
                            cx1  = (lm1[4].x + lm1[8].x) / 2
                            cy1  = (lm1[4].y + lm1[8].y) / 2
                            dist = math.sqrt((cx0 - cx1)**2 + (cy0 - cy1)**2)
                            ctrl.update_zoom(True, dist, now)
                        else:
                            ctrl.update_zoom(False, 0.0, now)

                            fingers1  = fingers_open(lm1, label1)
                            both_open = sum(fingers) >= 4 and sum(fingers1) >= 4
                            if both_open:
                                if select_hold_t == 0.0:
                                    select_hold_t = now
                                elif (now - select_hold_t >= 1.0
                                        and autoframe_on and autoframer is not None
                                        and len(last_faces) > 0
                                        and now - last_select_t > 2.0):
                                    crp = autoframer.get_crop()
                                    cw  = crp[2] - crp[0]
                                    ch  = crp[3] - crp[1]
                                    hx  = (palm_pos(lm)[0] + palm_pos(lm1)[0]) / 2
                                    hy  = (palm_pos(lm)[1] + palm_pos(lm1)[1]) / 2
                                    hand_nx = (hx * cw + crp[0]) / cap.width
                                    hand_ny = (hy * ch + crp[1]) / cap.height
                                    sel     = _choose_face(last_faces, (hand_nx, hand_ny),
                                                           cap.width, cap.height)
                                    inv_s   = 1.0 / FACE_DET_SCALE
                                    preferred_face = (
                                        (sel[0] + sel[2] / 2) * inv_s / cap.width,
                                        (sel[1] + sel[3] / 2) * inv_s / cap.height,
                                    )
                                    last_select_t = now
                                    select_hold_t = 0.0
                                    ctrl._flash("SELECCIONADO")
                            else:
                                select_hold_t = 0.0

                            if is_pointing(lm, hand_label) and is_ok_sign(lm1, label1):
                                combo = hands_united(lm, lm1)
                            elif is_ok_sign(lm, hand_label) and is_pointing(lm1, label1):
                                combo = hands_united(lm1, lm)
                            else:
                                combo = False
                            if combo and now - last_sound_t > SOUND_COOLDOWN:
                                play_sound(SOUND_SEXY)
                                last_sound_t = now
                    else:
                        overlay.update(px, py, True)
                        ctrl.update_zoom(False, 0.0, now)
                else:
                    ctrl.no_hand(now)
                    overlay.update(0.5, 0.5, False)
                    if not headless:
                        draw_hud(frame_mp if autoframe_on else frame, ctrl, [False] * 5, False, False, False, now)
            except pyautogui.FailSafeException:
                ctrl.cleanup()
                ctrl.no_hand(now)
                overlay.update(0.5, 0.5, False)

            if headless:
                if stop_flag.is_set():
                    break
            else:
                if autoframe_on:
                    display = cv2.resize(frame_mp, (cap.width, cap.height))
                else:
                    display = frame
                if DISPLAY_SCALE != 1.0:
                    dw = int(cap.width  * DISPLAY_SCALE)
                    dh = int(cap.height * DISPLAY_SCALE)
                    display = cv2.resize(display, (dw, dh))
                bar_w = display.shape[1]
                hud   = np.zeros((HUD_BAR_H, bar_w, 3), np.uint8)
                cv2.putText(hud, ctrl.status, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 210, 210), 1, cv2.LINE_AA)
                for i, line in enumerate(_HUD_LEGENDS):
                    x = 10 if i < 3 else bar_w // 2
                    y = 46 + (i % 3) * 18
                    cv2.putText(hud, line, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (130, 130, 130), 1, cv2.LINE_AA)
                cv2.imshow("Controlador de Mano  |  ESC = salir",
                           np.vstack([display, hud]))
                if cv2.waitKey(1) & 0xFF == 27 or stop_flag.is_set():
                    break

    ctrl.cleanup()
    mover.stop()
    overlay.close()
    if tray:
        tray.stop()
    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print("Cerrando.")
    hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    if hwnd:
        ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE


if __name__ == "__main__":
    main()
