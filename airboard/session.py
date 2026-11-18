from .core import Drawing, PalmToggle


class Session:
    def __init__(self):
        self.drawing = Drawing()
        self.palm = PalmToggle()
        self.board = False
        self.cursor = None
        self.draw_since = None

    def update(self, authorized, gesture, tip, now, hand_count=1):
        self.cursor = None
        allowed = authorized and hand_count == 1
        if self.palm.update(gesture == "palm", now, allowed):
            self.board = not self.board
            self.drawing.lift()
        if not (allowed and self.board and gesture == "draw" and tip is not None):
            self.draw_since = None
            self.drawing.lift()
            return
        if self.draw_since is None:
            self.draw_since = now
        # Central camera rectangle maps to the entire canvas.
        x, y = ((tip[0] - 0.15) / 0.70, (tip[1] - 0.15) / 0.70)
        if not (0 <= x <= 1 and 0 <= y <= 1):
            self.drawing.lift()
            return
        self.cursor = (x, y)
        if now - self.draw_since >= 0.12:
            self.drawing.add((x, y))
