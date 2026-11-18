#!/usr/bin/env python3
"""Run from the local virtual environment. See README.md."""

import argparse
from pathlib import Path
import sys
import time
from airboard.models import ROOT, download


def parser():
    p = argparse.ArgumentParser(
        description="AirBoard — персональная бесконтактная доска"
    )
    p.add_argument(
        "--person", type=Path, default=ROOT / "person.jpg", help="Фото с одним лицом"
    )
    p.add_argument("--camera", type=int, default=0, help="Индекс камеры (обычно 0)")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Порог cosine similarity, не вероятность (по умолчанию .45)",
    )
    p.add_argument(
        "--download-models",
        action="store_true",
        help="Загрузить и проверить публичные модели, без камеры",
    )
    p.add_argument(
        "--check", action="store_true", help="Проверить модели и фото, без камеры"
    )
    p.add_argument(
        "--preview",
        type=Path,
        help="Создать синтетический preview PNG без камеры и фото",
    )
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0.1 <= args.threshold <= 1:
        raise ValueError("--threshold должен быть от .1 до 1")
    if args.camera < 0:
        raise ValueError("--camera должен быть неотрицательным")
    if args.download_models:
        download()
        return 0
    import cv2
    import numpy as np
    from airboard.session import Session
    from airboard.ui import render, export

    session = Session()
    if args.preview:
        session.board = True
        for start, end in (((0.12, 0.65), (0.18, 0.30)), ((0.18, 0.30), (0.25, 0.65))):
            for t in np.linspace(0, 1, 50):
                session.drawing.add(
                    tuple(a + float(t) * (b - a) for a, b in zip(start, end))
                )
        session.drawing.lift()
        for point in ((0.14, 0.53), (0.22, 0.53)):
            session.drawing.add(point)
        session.drawing.lift()
        for x in np.linspace(0.35, 0.85, 150):
            session.drawing.add((float(x), float(0.49 + 0.13 * np.sin(x * 22))))
        camera = np.full((360, 640, 3), (42, 35, 28), np.uint8)
        cv2.putText(
            camera,
            "CAMERA PREVIEW",
            (110, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (180, 170, 150),
            1,
        )
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(args.preview),
            render(
                session,
                camera,
                True,
                "PREVIEW - SYNTHETIC",
                "draw",
                notice="Synthetic preview. No camera or face recognition was used.",
            ),
        ):
            raise OSError("Cannot write preview")
        print(args.preview)
        return 0
    if not args.person.is_file():
        raise ValueError(f"Добавь своё фото с одним лицом: {args.person}")
    from airboard.vision import FaceMatcher, HandTracker

    matcher = FaceMatcher(args.person, args.threshold)
    if args.check:
        tracker = HandTracker()
        try:
            tracker.check(np.zeros((360, 640, 3), np.uint8))
        finally:
            tracker.close()
        print("OK: face models, reference photo, MediaPipe Hands. Camera not opened.")
        return 0
    capture = None
    tracker = None
    try:
        tracker = HandTracker()
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        capture = cv2.VideoCapture(args.camera, backend)
        if not capture.isOpened():
            raise RuntimeError(
                "Камера недоступна. Разреши доступ к камере для Terminal/Python в настройках macOS, закрой другие приложения или попробуй --camera 1."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, 30)
        window = "AirBoard"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 800)
        notice, notice_until, clear_until = "", 0.0, 0.0
        fullscreen = False
        previous, fps = time.monotonic(), 0.0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    "Поток камеры прервался. Проверь подключение и перезапусти приложение."
                )
            frame = FaceMatcher.resize(frame, 640)
            authorized, status, score = matcher.check(frame)
            mirrored = cv2.flip(frame, 1)
            gesture, points, count = tracker.check(mirrored)
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(0.001, now - previous)
            previous = now
            h, w = mirrored.shape[:2]
            tip = (points[8][0] / w, points[8][1] / h) if points else None
            session.update(authorized, gesture, tip, now, count)
            if count > 1 and authorized:
                status = "SHOW ONE HAND"
            cv2.rectangle(
                mirrored,
                (int(w * 0.15), int(h * 0.15)),
                (int(w * 0.85), int(h * 0.85)),
                (120, 140, 100),
                1,
            )
            if points:
                cv2.circle(mirrored, tuple(map(int, points[8])), 6, (255, 210, 55), 2)
            image = render(
                session,
                mirrored,
                authorized,
                status,
                gesture,
                score,
                fps,
                notice if now < notice_until else "",
            )
            cv2.imshow(window, image)
            key = cv2.waitKey(1) & 0xFF
            if (
                key in (27, ord("q"))
                or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
            ):
                break
            if key == ord("c"):
                session.drawing.next_color()
            elif key == ord("z"):
                session.drawing.undo()
            elif key == ord("x"):
                if now < clear_until:
                    session.drawing.clear()
                    notice, clear_until = "Board cleared", 0.0
                else:
                    notice, clear_until = (
                        "Press X again within 2 seconds to clear",
                        now + 2,
                    )
                notice_until = now + 2
            elif key == ord("s"):
                try:
                    path = export(session.drawing, ROOT / "exports")
                    notice = f"Saved: exports/{path.name}"
                    print(f"Saved: {path}", flush=True)
                except OSError:
                    notice = "Save failed: check disk space and folder permissions"
                notice_until = now + 4
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    window,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
    finally:
        if capture is not None:
            capture.release()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"AirBoard: {exc}", file=sys.stderr)
        raise SystemExit(1)
