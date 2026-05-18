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
import argparse
from typing import Optional, Tuple


# Movimiento de mouse directo via Win32 API (sin overhead de pyautogui)
def _move_cursor(x: int, y: int):
    ctypes.windll.user32.SetCursorPos(x, y)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

# ═══════════════════════════ CONFIGURACIÓN ════════════════════════════
CAMERA_ID  = 0
FRAME_W    = 640
FRAME_H    = 480

SCREEN_W, SCREEN_H = pyautogui.size()

# One Euro Filter — reduce jitter sin agregar lag
OEF_MINCUTOFF = 3.5    # menor = más suavizado cuando la mano está quieta
OEF_BETA      = 1.8    # mayor = más responsivo cuando la mano se mueve rápido
MARGIN              = 0.12
FIST_MAX_OPEN       = 1
CLICK_MAX_HOLD      = 0.35
DRAG_START_HOLD     = 0.45
DOUBLE_FIST_WINDOW  = 0.50
SCROLL_SPEED_FACTOR = 3000  # scrolls/segundo por unidad^2 de offset (escala cuadratica)
SCROLL_DEADZONE     = 0.04  # zona muerta central antes de empezar a scrollear
PINCH_THRESHOLD     = 0.06  # distancia normalizada pulgar-indice para detectar pinch
DRAG_PINCH_HOLD     = 0.5   # segundos manteniendo pinch para activar arrastre
ZOOM_SENSITIVITY    = 50    # clicks de zoom por unidad de distancia entre manos
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
    def __init__(self, camera_id: int, width: int, height: int):
        self._cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimiza frames en buffer
        self._frame: Optional[np.ndarray] = None
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

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

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
    def __init__(self):
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
        self.status = "Sin mano detectada"

    def cleanup(self):
        if self._dragging:
            pyautogui.mouseUp()
        if self._zoom_active:
            pyautogui.keyUp('ctrl')
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
            _move_cursor(mx, my)
            self.status = f"Arrastrando ({mx}, {my})"
        elif is_pinch:
            held = now - self._pinch_start
            _move_cursor(mx, my)
            self.status = f"Pinch {held:.1f}s / {DRAG_PINCH_HOLD:.0f}s"
        else:
            _move_cursor(mx, my)
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

    @property
    def pinching(self) -> bool:
        return self._pinch_active


# ──────────────────────────── Overlay ────────────────────────────────

class HandOverlay:
    """Ventana translucida siempre encima que muestra la posicion de la mano."""
    SIZE = 180

    def __init__(self):
        self._q     = queue.Queue(maxsize=2)
        self._alive = True
        threading.Thread(target=self._run, daemon=True).start()

    def update(self, nx: float, ny: float, visible: bool,
               nx2: float = 0.5, ny2: float = 0.5, visible2: bool = False):
        try:
            self._q.put_nowait((nx, ny, visible, nx2, ny2, visible2))
        except queue.Full:
            pass

    def close(self):
        self._alive = False

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
                return
            try:
                while True:
                    nx_v[0], ny_v[0], vis_v[0], nx2_v[0], ny2_v[0], vis2_v[0] = self._q.get_nowait()
            except queue.Empty:
                pass

            cv.delete("all")

            # fondo
            cv.create_rectangle(0, 0, S, S, fill="#0a0a1a", outline="")
            # zona activa del mouse (sin margen)
            cv.create_rectangle(M, M, S - M, S - M, fill="", outline=ZONE, width=1)
            # borde exterior con esquinas marcadas
            cv.create_rectangle(1, 1, S - 1, S - 1, outline=EDGE, width=2)
            # esquinas reforzadas
            c = 10
            for x0, y0, x1, y1 in [(1,1,c,1),(1,1,1,c),(S-c,1,S-1,1),(S-1,1,S-1,c),
                                     (1,S-c,1,S-1),(1,S-1,c,S-1),(S-c,S-1,S-1,S-1),(S-1,S-c,S-1,S-1)]:
                cv.create_line(x0, y0, x1, y1, fill=EDGE, width=3)

            if vis_v[0]:
                dx = int(nx_v[0] * S)
                dy = int(ny_v[0] * S)
                r  = 7
                cv.create_oval(dx-r+1, dy-r+1, dx+r+1, dy+r+1, fill="#003333", outline="")
                cv.create_oval(dx-r, dy-r, dx+r, dy+r, fill=EDGE, outline="")

            if vis2_v[0]:
                dx2 = int(nx2_v[0] * S)
                dy2 = int(ny2_v[0] * S)
                r   = 7
                cv.create_oval(dx2-r+1, dy2-r+1, dx2+r+1, dy2+r+1, fill="#331a00", outline="")
                cv.create_oval(dx2-r, dy2-r, dx2+r, dy2+r, fill=EDGE2, outline="")

            root.after(30, redraw)

        redraw()
        root.mainloop()


# ──────────────────────────── HUD ─────────────────────────────────────

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

    cv2.rectangle(frame, (0, h - 42), (w, h), (15, 15, 15), -1)
    cv2.putText(frame, ctrl.status, (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 210, 210), 1, cv2.LINE_AA)

    gesture = ctrl.gesture_overlay
    if gesture:
        tw = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)[0][0]
        alpha = min(1.0, (ctrl._gesture_until - now) / 0.25)
        col   = (int(255 * alpha), int(255 * alpha), 50)
        cv2.putText(frame, gesture, ((w - tw) // 2, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, col, 2, cv2.LINE_AA)

    legends = [
        "Mano abierta -> mover",
        "Pinch        -> click izq",
        "Pinch 0.5s   -> arrastrar",
        "Puno         -> scroll V+H",
        "2x Pinch     -> zoom",
        "V (paz)      -> Peron",
    ]
    for i, line in enumerate(legends):
        cv2.putText(frame, line, (w - 215, h - 100 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1, cv2.LINE_AA)


# ──────────────────────────── Tray icon (Win32 puro) ──────────────────

class _TrayIcon:
    """Icono en bandeja del sistema via Win32 API. Sin dependencias extra."""
    _WM_TRAY = 0x8001
    _ID_QUIT = 9001

    def __init__(self, stop_event: threading.Event):
        self._stop        = stop_event
        self._hwnd        = None
        self._wndproc_ref = None   # evitar GC del callback

    def run(self):
        """Llamar desde un hilo daemon."""
        u32     = ctypes.windll.user32
        shell32 = ctypes.windll.shell32
        k32     = ctypes.windll.kernel32

        WM_DESTROY    = 0x0002
        WM_RBUTTONUP  = 0x0205
        NIM_ADD       = 0
        NIM_DELETE    = 2
        NIF_MESSAGE   = 0x1
        NIF_ICON      = 0x2
        NIF_TIP       = 0x4
        TPM_RETURNCMD = 0x0100
        TPM_NONOTIFY  = 0x0080
        MF_STRING     = 0x0000

        class _NID(ctypes.Structure):
            _fields_ = [
                ("cbSize",           ctypes.c_ulong),
                ("hWnd",             ctypes.c_void_p),
                ("uID",              ctypes.c_uint),
                ("uFlags",           ctypes.c_uint),
                ("uCallbackMessage", ctypes.c_uint),
                ("hIcon",            ctypes.c_void_p),
                ("szTip",            ctypes.c_wchar * 128),
            ]

        class _WNDCLASSEX(ctypes.Structure):
            _fields_ = [
                ("cbSize",        ctypes.c_uint),
                ("style",         ctypes.c_uint),
                ("lpfnWndProc",   ctypes.c_void_p),
                ("cbClsExtra",    ctypes.c_int),
                ("cbWndExtra",    ctypes.c_int),
                ("hInstance",     ctypes.c_void_p),
                ("hIcon",         ctypes.c_void_p),
                ("hCursor",       ctypes.c_void_p),
                ("hbrBackground", ctypes.c_void_p),
                ("lpszMenuName",  ctypes.c_wchar_p),
                ("lpszClassName", ctypes.c_wchar_p),
                ("hIconSm",       ctypes.c_void_p),
            ]

        class _MSG(ctypes.Structure):
            _fields_ = [
                ("hwnd",    ctypes.c_void_p),
                ("message", ctypes.c_uint),
                ("wParam",  ctypes.c_uint),
                ("lParam",  ctypes.c_long),
                ("time",    ctypes.c_ulong),
                ("pt",      ctypes.c_long * 2),
            ]

        WNDPROC = ctypes.WINFUNCTYPE(
            ctypes.c_long,
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_long,
        )

        def wnd_proc(hwnd, msg, wp, lp):
            if msg == self._WM_TRAY and lp == WM_RBUTTONUP:
                hmenu = u32.CreatePopupMenu()
                u32.AppendMenuW(hmenu, MF_STRING, self._ID_QUIT, "Cerrar Hand Controller")
                pt = ctypes.wintypes.POINT()
                u32.GetCursorPos(ctypes.byref(pt))
                u32.SetForegroundWindow(hwnd)
                cmd = u32.TrackPopupMenu(
                    hmenu, TPM_RETURNCMD | TPM_NONOTIFY,
                    pt.x, pt.y, 0, hwnd, None,
                )
                u32.DestroyMenu(hmenu)
                if cmd == self._ID_QUIT:
                    u32.PostMessageW(hwnd, WM_DESTROY, 0, 0)
                return 0
            if msg == WM_DESTROY:
                nid = _NID()
                nid.cbSize = ctypes.sizeof(_NID)
                nid.hWnd   = hwnd
                nid.uID    = 1
                shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(nid))
                u32.PostQuitMessage(0)
                self._stop.set()
                return 0
            return u32.DefWindowProcW(hwnd, msg, wp, lp)

        self._wndproc_ref = WNDPROC(wnd_proc)
        hinstance = k32.GetModuleHandleW(None)
        cls_name  = "HandCtrlTray"

        wc = _WNDCLASSEX()
        wc.cbSize        = ctypes.sizeof(_WNDCLASSEX)
        wc.lpfnWndProc   = ctypes.cast(self._wndproc_ref, ctypes.c_void_p)
        wc.hInstance     = hinstance
        wc.lpszClassName = cls_name
        u32.RegisterClassExW(ctypes.byref(wc))

        hwnd = u32.CreateWindowExW(
            0, cls_name, "Hand Controller",
            0, 0, 0, 0, 0, -3,   # -3 = HWND_MESSAGE
            None, hinstance, None,
        )
        self._hwnd = hwnd

        hicon = u32.LoadIconW(None, 32512)  # IDI_APPLICATION

        nid = _NID()
        nid.cbSize           = ctypes.sizeof(_NID)
        nid.hWnd             = hwnd
        nid.uID              = 1
        nid.uFlags           = NIF_MESSAGE | NIF_ICON | NIF_TIP
        nid.uCallbackMessage = self._WM_TRAY
        nid.hIcon            = hicon
        nid.szTip            = "Hand Controller"
        shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(nid))

        msg = _MSG()
        while u32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            u32.TranslateMessage(ctypes.byref(msg))
            u32.DispatchMessageW(ctypes.byref(msg))

        u32.UnregisterClassW(cls_name, hinstance)

    def stop(self):
        if self._hwnd:
            ctypes.windll.user32.PostMessageW(self._hwnd, 0x0002, 0, 0)


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

    cap = CameraCapture(CAMERA_ID, FRAME_W, FRAME_H)

    if not cap.opened:
        print(f"ERROR: No se pudo abrir la camara {CAMERA_ID}")
        return

    ctrl    = HandController()
    overlay = HandOverlay()

    if headless:
        tray = _TrayIcon(stop_flag)
        threading.Thread(target=tray.run, daemon=True).start()
    else:
        tray = None

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
        min_hand_detection_confidence=0.72,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.65,
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

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.opened:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.001)
                continue

            frame = cv2.flip(frame, 1)
            now   = time.time()

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms    = int((now - t0) * 1000)

            # No bloqueante: el resultado llega via on_result en otro hilo
            landmarker.detect_async(mp_image, ts_ms)

            with _lock:
                result = _latest[0]

            if result and result.hand_landmarks and result.handedness:
                lm         = result.hand_landmarks[0]
                # Tasks API reporta la mano desde perspectiva del modelo (imagen sin flipear)
                # como flipemos el frame, invertimos la etiqueta para que coincida con
                # la mano real del usuario
                raw_label  = result.handedness[0][0].category_name
                hand_label = "Left" if raw_label == "Right" else "Right"

                mx, my, fingers, is_fist, is_pinch, is_peace = ctrl.process(lm, hand_label, now)
                px, py = palm_pos(lm)
                if not headless:
                    draw_hand(frame, lm, is_pinch)
                    draw_hud(frame, ctrl, fingers, is_fist, is_pinch, is_peace, now)

                # Gestos de dos manos
                if len(result.hand_landmarks) >= 2:
                    lm1    = result.hand_landmarks[1]
                    raw1   = result.handedness[1][0].category_name
                    label1 = "Left" if raw1 == "Right" else "Right"
                    px1, py1 = palm_pos(lm1)
                    overlay.update(px, py, True, px1, py1, True)
                    if not headless:
                        draw_hand(frame, lm1)

                    # Doble pinch → zoom (Ctrl + scroll)
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
                        # Combo: 👉 + 👌
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
                    draw_hud(frame, ctrl, [False] * 5, False, False, False, now)

            if headless:
                if stop_flag.is_set():
                    break
            else:
                cv2.imshow("Controlador de Mano  |  ESC = salir", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    ctrl.cleanup()
    overlay.close()
    if tray:
        tray.stop()
    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print("Cerrando.")


if __name__ == "__main__":
    main()
