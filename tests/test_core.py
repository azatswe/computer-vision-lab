import math
import unittest
from airboard.core import classify, joint_angle, IdentityGate, PalmToggle, Drawing
from airboard.session import Session


def hand(extended=(True, True, True, True), thumb=True):
    p = [(0.0, 0.0)] * 21
    p[0] = (0, 5)
    p[1:5] = (
        [(-1, 4), (-2, 3), (-3, 2), (-4, 1)]
        if thumb
        else [(-1, 4), (-2, 3), (-1, 3), (0, 3)]
    )
    for x, base, straight in zip((-1.5, -0.5, 0.5, 1.5), (5, 9, 13, 17), extended):
        p[base : base + 4] = (
            [(x, 3), (x, 2), (x, 1), (x, 0)]
            if straight
            else [(x, 3), (x, 2), (x + 0.3, 3), (x + 0.4, 3.5)]
        )
    return p


class GestureTests(unittest.TestCase):
    def test_palm(self):
        self.assertEqual(classify(hand()), "palm")

    def test_index(self):
        self.assertEqual(classify(hand((True, False, False, False), False)), "draw")

    def test_fist(self):
        self.assertEqual(classify(hand((False,) * 4, False)), "idle")

    def test_two_fingers(self):
        self.assertEqual(classify(hand((True, True, False, False), False)), "idle")

    def test_four_fingers_without_thumb_not_palm(self):
        self.assertEqual(classify(hand(thumb=False)), "idle")

    def test_bad_landmarks(self):
        self.assertEqual(classify([]), "idle")

    def test_degenerate(self):
        self.assertEqual(classify([(0, 0)] * 21), "idle")

    def test_angle(self):
        self.assertEqual(joint_angle((0, 0), (1, 0), (2, 0)), 180)

    def test_rotation_and_mirror(self):
        for angle in (0, 0.4, 1.5, 3):
            for mirror in (-1, 1):
                p = [
                    (
                        mirror * (x * math.cos(angle) - y * math.sin(angle)),
                        x * math.sin(angle) + y * math.cos(angle),
                    )
                    for x, y in hand()
                ]
                self.assertEqual(classify(p), "palm")


class StateTests(unittest.TestCase):
    def test_identity_requires_stability_and_revokes_immediately(self):
        gate = IdentityGate()
        self.assertEqual(
            [gate.update(x) for x in (True, True, True, False, True)],
            [False, False, True, False, False],
        )

    def test_hold_one_shot_release_and_cooldown(self):
        p = PalmToggle()
        self.assertFalse(p.update(True, 0))
        self.assertFalse(p.update(True, 0.4))
        self.assertTrue(p.update(True, 0.81))
        self.assertFalse(p.update(True, 5))
        p.update(False, 5.1)
        p.update(False, 5.5)
        self.assertFalse(p.update(True, 5.6))
        self.assertTrue(p.update(True, 6.5))

    def test_authorization_interrupts_hold(self):
        p = PalmToggle()
        p.update(True, 0)
        p.update(True, 0.7, False)
        self.assertFalse(p.update(True, 0.9))
        self.assertTrue(p.update(True, 1.8))

    def test_short_palm_is_ignored(self):
        p = PalmToggle()
        p.update(True, 0)
        p.update(False, 0.5)
        self.assertFalse(p.update(True, 1))

    def test_no_drawing_on_camera(self):
        s = Session()
        s.update(True, "draw", (0.5, 0.5), 0)
        s.update(True, "draw", (0.5, 0.5), 1)
        self.assertFalse(s.drawing.strokes)

    def test_toggle_preserves_strokes(self):
        s = Session()
        s.drawing.add((0.2, 0.3))
        s.update(True, "palm", None, 0)
        s.update(True, "palm", None, 1)
        self.assertTrue(s.board)
        self.assertEqual(len(s.drawing.strokes), 1)

    def test_identity_loss_lifts_pen(self):
        s = Session()
        s.board = True
        s.update(True, "draw", (0.5, 0.5), 0)
        s.update(True, "draw", (0.51, 0.5), 0.2)
        self.assertEqual(len(s.drawing.strokes), 1)
        s.update(False, "draw", (0.53, 0.5), 0.3)
        self.assertIsNone(s.drawing.active)
        s.update(True, "draw", (0.53, 0.5), 0.4)
        s.update(True, "draw", (0.53, 0.5), 0.6)
        self.assertEqual(len(s.drawing.strokes), 2)

    def test_multiple_hands_cannot_toggle_or_draw(self):
        s = Session()
        s.board = True
        s.update(True, "draw", (0.5, 0.5), 0, 2)
        s.update(True, "draw", (0.5, 0.5), 1, 2)
        s.update(True, "palm", None, 2, 2)
        s.update(True, "palm", None, 3, 2)
        self.assertTrue(s.board)
        self.assertFalse(s.drawing.strokes)

    def test_outside_guide_lifts_pen(self):
        s = Session()
        s.board = True
        s.update(True, "draw", (0.5, 0.5), 0)
        s.update(True, "draw", (0.5, 0.5), 0.2)
        s.update(True, "draw", (0.99, 0.5), 0.3)
        self.assertIsNone(s.drawing.active)

    def test_fist_lifts_pen(self):
        s = Session()
        s.board = True
        s.update(True, "draw", (0.5, 0.5), 0)
        s.update(True, "draw", (0.5, 0.5), 0.2)
        s.update(True, "idle", None, 0.3)
        self.assertIsNone(s.drawing.active)

    def test_stroke_boundary_color_undo_clear(self):
        d = Drawing()
        d.add((0.1, 0.1))
        d.add((0.11, 0.11))
        d.next_color()
        d.add((0.12, 0.12))
        self.assertEqual(len(d.strokes), 2)
        self.assertNotEqual(d.strokes[0].color, d.strokes[1].color)
        d.undo()
        self.assertEqual(d.total, 2)
        d.clear()
        self.assertEqual(d.total, 0)
        d.undo()
        self.assertEqual(d.total, 0)

    def test_tracking_jump_does_not_connect(self):
        d = Drawing()
        d.add((0.1, 0.1))
        d.add((0.9, 0.9))
        self.assertEqual(len(d.strokes), 2)

    def test_memory_bound(self):
        d = Drawing(max_points=2)
        d.add((0.1, 0.1))
        d.add((0.12, 0.12))
        d.add((0.14, 0.14))
        self.assertTrue(d.full)
        self.assertEqual(d.total, 2)
        d.undo()
        self.assertFalse(d.full)


if __name__ == "__main__":
    unittest.main()
