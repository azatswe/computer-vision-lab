"""Pure application logic, independent of camera and ML libraries."""

from dataclasses import dataclass, field
from math import acos, degrees, hypot


def distance(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])


def joint_angle(a, b, c):
    u = (a[0] - b[0], a[1] - b[1])
    v = (c[0] - b[0], c[1] - b[1])
    length = hypot(*u) * hypot(*v)
    if length < 1e-9:
        return 0.0
    return degrees(acos(max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / length))))


def classify(points):
    """21 pixel landmarks; rotation/mirror-independent, conservative gestures."""
    if len(points) != 21:
        return "idle"
    fingers = []
    for mcp, pip, dip, tip in (
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ):
        straight = (
            joint_angle(points[mcp], points[pip], points[tip]) > 155
            and joint_angle(points[pip], points[dip], points[tip]) > 150
            and distance(points[tip], points[0])
            > 1.18 * distance(points[pip], points[0])
        )
        fingers.append(straight)
    thumb = joint_angle(points[1], points[2], points[4]) > 145 and distance(
        points[4], points[5]
    ) > 0.65 * distance(points[5], points[17])
    if all(fingers) and thumb:
        return "palm"
    if fingers == [True, False, False, False]:
        return "draw"
    return "idle"


@dataclass
class IdentityGate:
    required: int = 3
    hits: int = 0

    def update(self, matches):
        self.hits = min(self.required, self.hits + 1) if matches else 0
        return self.hits >= self.required


@dataclass
class PalmToggle:
    hold: float = 0.8
    release: float = 0.35
    cooldown: float = 1.5
    started: float | None = None
    released: float | None = None
    armed: bool = True
    last: float = -100.0
    progress: float = 0.0

    def update(self, palm, now, authorized=True):
        self.progress = 0.0
        if not authorized:
            self.started = self.released = None
            return False
        if not palm:
            self.started = None
            if self.released is None:
                self.released = now
            if now - self.released >= self.release:
                self.armed = True
            return False
        self.released = None
        if not self.armed or now - self.last < self.cooldown:
            return False
        if self.started is None:
            self.started = now
        self.progress = min(1.0, (now - self.started) / self.hold)
        if self.progress >= 1:
            self.armed = False
            self.last = now
            self.started = None
            return True
        return False


PALETTE = ((255, 210, 55), (130, 240, 120), (100, 125, 255), (235, 235, 245))  # BGR


@dataclass
class Stroke:
    color: tuple
    points: list = field(default_factory=list)


class Drawing:
    def __init__(self, max_points=40000):
        self.strokes = []
        self.active = None
        self.color = 0
        self.max_points = max_points
        self.total = 0
        self.full = False

    def lift(self):
        self.active = None

    def add(self, point):
        if self.total >= self.max_points:
            self.full = True
            self.lift()
            return
        point = tuple(map(float, point))
        if self.active and distance(point, self.active.points[-1]) > 0.20:
            self.lift()  # Tracking jump must not draw across the screen.
        if self.active is None:
            self.active = Stroke(PALETTE[self.color], [point])
            self.strokes.append(self.active)
            self.total += 1
        else:
            old = self.active.points[-1]
            smooth = tuple(a + 0.40 * (b - a) for a, b in zip(old, point))
            if distance(old, smooth) >= 0.0015:
                self.active.points.append(smooth)
                self.total += 1

    def undo(self):
        self.lift()
        if self.strokes:
            self.total -= len(self.strokes.pop().points)
        self.full = False

    def clear(self):
        self.strokes.clear()
        self.total = 0
        self.full = False
        self.lift()

    def next_color(self):
        self.color = (self.color + 1) % len(PALETTE)
        self.lift()
