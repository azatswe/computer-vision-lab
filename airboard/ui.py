from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from .core import PALETTE

WIDTH, HEIGHT = 1280, 800
BX, BY, BW, BH = 24, 130, 952, 536
BG = (23, 19, 16)
MUTED = (155, 148, 137)
WHITE = (245, 242, 233)
ACCENT = (255, 210, 55)


def text(image, value, x, y, scale=0.55, color=WHITE, thickness=1):
    cv2.putText(
        image,
        value,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def fit(image, width, height):
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
    output = np.full((height, width, 3), BG, np.uint8)
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    output[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return output


def canvas(drawing, width=BW, height=BH, grid=True):
    image = np.full((height, width, 3), (32, 28, 24), np.uint8)
    if grid:
        for y in range(20, height, 28):
            for x in range(20, width, 28):
                image[y, x] = (63, 55, 45)
    for stroke in drawing.strokes:
        pts = np.array(
            [
                (round(x * (width - 1)), round(y * (height - 1)))
                for x, y in stroke.points
            ],
            np.int32,
        )
        if len(pts) == 1:
            cv2.circle(image, tuple(pts[0]), 3, stroke.color, -1, cv2.LINE_AA)
        elif len(pts) > 1:
            cv2.polylines(image, [pts], False, stroke.color, 4, cv2.LINE_AA)
    return image


def export(drawing, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        "airboard-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".png"
    )
    # Only the drawing: no face, camera frame, similarity score or HUD.
    if not cv2.imwrite(str(path), canvas(drawing, 1600, 900, grid=False)):
        raise OSError("Не удалось сохранить PNG")
    return path


def render(
    session,
    camera,
    authorized=False,
    status="LOOK AT CAMERA",
    gesture="idle",
    score=None,
    fps=0,
    notice="",
):
    image = np.full((HEIGHT, WIDTH, 3), BG, np.uint8)
    text(image, "AIRBOARD", 26, 48, 1.0, WHITE, 2)
    text(image, "PERSONAL SPACE FOR BIG IDEAS", 26, 78, 0.42, MUTED)
    text(image, "LOCAL ONLY  /  CAMERA NOT RECORDED", 870, 45, 0.43, MUTED)
    cv2.line(image, (24, 96), (1256, 96), (65, 54, 43), 1)
    text(image, "BOARD" if session.board else "CAMERA", BX, 119, 0.45, ACCENT)
    if session.board:
        image[BY : BY + BH, BX : BX + BW] = canvas(session.drawing)
        if not session.drawing.strokes:
            text(image, "Your next idea starts here.", BX + 190, BY + 235, 0.9, MUTED)
            text(
                image,
                "Raise only your index finger to draw.",
                BX + 205,
                BY + 275,
                0.57,
                MUTED,
            )
        if session.cursor:
            x, y = session.cursor
            cv2.circle(
                image,
                (BX + int(x * (BW - 1)), BY + int(y * (BH - 1))),
                10,
                PALETTE[session.drawing.color],
                2,
                cv2.LINE_AA,
            )
    else:
        image[BY : BY + BH, BX : BX + BW] = fit(camera, BW, BH)
        text(
            image,
            "Hold an open palm to open your board",
            BX + 110,
            BY + BH - 28,
            0.7,
            ACCENT,
            2,
        )
    cv2.rectangle(image, (BX, BY), (BX + BW, BY + BH), (65, 54, 43), 1)
    image[130:274, 1000:1256] = fit(camera, 256, 144)
    cv2.circle(
        image, (1010, 310), 5, (100, 220, 140) if authorized else (100, 160, 255), -1
    )
    text(image, status, 1026, 315, 0.42)
    if score is not None:
        text(image, f"Match score: {score:.2f}", 1000, 344, 0.42, MUTED)
    text(image, "GESTURE CONTROLS", 1000, 390, 0.45, ACCENT)
    text(image, "Open palm / hold 0.8s", 1000, 421, 0.45)
    text(image, "Switch camera / board", 1000, 444, 0.40, MUTED)
    text(image, "Index finger only", 1000, 480, 0.45)
    text(image, "Draw on your board", 1000, 503, 0.40, MUTED)
    text(image, "Other gestures: pen up", 1000, 539, 0.42, MUTED)
    text(image, "One person, one hand", 1000, 576, 0.43, MUTED)
    text(image, f"{gesture.upper()}  /  {fps:.0f} FPS", 1000, 624, 0.43, ACCENT)
    if session.palm.progress:
        cv2.rectangle(
            image,
            (1000, 644),
            (1000 + int(256 * session.palm.progress), 650),
            ACCENT,
            -1,
        )
    for i, color in enumerate(PALETTE):
        cv2.circle(image, (40 + i * 35, 704), 9, color, -1, cv2.LINE_AA)
        if i == session.drawing.color:
            cv2.circle(image, (40 + i * 35, 704), 13, WHITE, 1, cv2.LINE_AA)
    text(
        image,
        "C color   Z undo   X clear (twice)   S save PNG   F fullscreen   Q quit",
        210,
        711,
        0.52,
    )
    text(
        image,
        notice
        or (
            "Canvas full: save, then clear or undo."
            if session.drawing.full
            else "Tip: keep your face visible and your hand inside the guide in the camera preview."
        ),
        26,
        758,
        0.48,
        MUTED,
    )
    return image
