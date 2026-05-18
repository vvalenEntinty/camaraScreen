# camaraScreen

Controlador de mouse por gestos de mano usando MediaPipe y OpenCV. Detecta gestos en tiempo real a través de la webcam y los traduce en acciones del sistema: mover el cursor, hacer click, scroll, arrastrar y zoom.

## Requisitos

- Python 3.9+
- Windows 10/11
- Webcam

## Instalación

```bash
pip install -r requirements.txt
```

El modelo de MediaPipe (~14 MB) se descarga automáticamente la primera vez que se ejecuta.

## Uso

**Modo normal** (con preview de cámara):
```bash
python hand_controller.py
```

**Modo headless** (sin ventana, solo icono en bandeja):
```bash
python hand_controller.py --headless
# o doble click en iniciar_headless.bat
```

Presionar `ESC` o click derecho en el icono de bandeja → *Cerrar* para salir.

## Gestos

| Gesto | Acción |
|-------|--------|
| Mano abierta | Mover mouse |
| Pinch (pulgar + índice) | Click izquierdo |
| Pinch sostenido 0.5s | Arrastrar |
| Puño cerrado | Scroll vertical y horizontal (según posición de la mano) |
| Doble pinch (dos manos) | Zoom (Ctrl + scroll) |
| ✌ (paz/victoria) | Música de Perón |

## Overlay

En la esquina inferior izquierda aparece una pequeña ventana que muestra la posición de las manos en tiempo real:
- Punto **verde azulado**: mano principal
- Punto **naranja**: segunda mano (cuando hay dos manos detectadas)
