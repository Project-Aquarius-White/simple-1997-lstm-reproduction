"""
Gradient Accumulation Animation

Visualizes how gradients accumulate separately for each LSTM component during backpropagation.
Shows three separate accumulators (Input, Input Gate, Output Gate) receiving gradients
from their respective components across time.

Scene: GradientAccumulation
Duration: ~35 seconds

Run: cd assets/animations && manimgl 03_gradient_accumulation.py GradientAccumulation -w
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manimlib import *
from colors import (
    CEC_BLUE, INPUT_GATE_GREEN, OUTPUT_GATE_ORANGE,
    CELL_INPUT_TEAL, GRADIENT_RED, TEXT_WHITE, ACCENT_YELLOW
)


class GradientAccumulation(InteractiveScene):
    """
    Animation showing gradient accumulation into 3 separate accumulators during LSTM backprop.
    """

    def construct(self):
        # ====================================================================
        # Title
        # ====================================================================
        title = Text(
            "Gradient Accumulation in LSTM",
            color=TEXT_WHITE,
            font_size=44
        )
        title.to_edge(UP, buff=0.5)

        self.play(Write(title), run_time=1.5)
        self.wait(0.5)

        # ====================================================================
        # Create 3 LSTM cells horizontally
        # Cell positions: t-2 (left), t-1 (center), t (right)
        # ====================================================================
        cell_width = 1.8
        cell_height = 2.0
        spacing = 2.5
        start_x = -2.5
        cell_y = 0.8

        cells = []
        time_labels = []

        timesteps = ["t-2", "t-1", "t"]

        for i, ts in enumerate(timesteps):
            # Create cell box
            cell = RoundedRectangle(
                width=cell_width,
                height=cell_height,
                corner_radius=0.15,
                color=CEC_BLUE,
                fill_color=CEC_BLUE,
                fill_opacity=0.2,
                stroke_width=2
            )
            x_pos = start_x + i * spacing
            cell.move_to([x_pos, cell_y, 0])
            cells.append(cell)

            # Time label above cell
            time_label = Tex(ts, color=TEXT_WHITE, font_size=24)
            time_label.next_to(cell, UP, buff=0.2)
            time_labels.append(time_label)

            self.play(FadeIn(cell), Write(time_label), run_time=0.8)

        self.wait(0.5)

        # ====================================================================
        # Component labels inside each cell (only 3 components)
        # ====================================================================
        components = [
            ("y_{in}", INPUT_GATE_GREEN, "input_gate"),
            ("y_{out}", OUTPUT_GATE_ORANGE, "output_gate"),
            ("g", CELL_INPUT_TEAL, "input")
        ]

        component_labels = [[] for _ in range(3)]

        for cell_idx, cell in enumerate(cells):
            for comp_idx, (comp_name, color, acc_type) in enumerate(components):
                label = Tex(
                    comp_name,
                    color=color,
                    font_size=20
                )
                # Position vertically within cell
                y_offset = 0.5 - comp_idx * 0.5
                label.move_to([cell.get_center()[0], cell_y + y_offset, 0])
                component_labels[cell_idx].append(label)
                self.play(FadeIn(label), run_time=0.3)

        self.wait(0.5)

        # ====================================================================
        # Create 3 separate gradient accumulator bars at bottom
        # ====================================================================
        acc_width = 2.0
        acc_height = 0.4
        acc_y = -2.0
        acc_spacing = 2.5
        acc_start_x = -2.5

        accumulators = {}
        acc_labels = {}

        # Input Gate accumulator (left)
        acc_ig = Rectangle(
            width=acc_width,
            height=acc_height,
            color=INPUT_GATE_GREEN,
            fill_color=INPUT_GATE_GREEN,
            fill_opacity=0.3,
            stroke_width=2
        )
        acc_ig.move_to([acc_start_x, acc_y, 0])
        accumulators['input_gate'] = acc_ig

        label_ig = Text("Input Gate", color=INPUT_GATE_GREEN, font_size=20)
        label_ig.next_to(acc_ig, DOWN, buff=0.15)
        acc_labels['input_gate'] = label_ig

        # Output Gate accumulator (center)
        acc_og = Rectangle(
            width=acc_width,
            height=acc_height,
            color=OUTPUT_GATE_ORANGE,
            fill_color=OUTPUT_GATE_ORANGE,
            fill_opacity=0.3,
            stroke_width=2
        )
        acc_og.move_to([0, acc_y, 0])
        accumulators['output_gate'] = acc_og

        label_og = Text("Output Gate", color=OUTPUT_GATE_ORANGE, font_size=20)
        label_og.next_to(acc_og, DOWN, buff=0.15)
        acc_labels['output_gate'] = label_og

        # Input accumulator (right)
        acc_in = Rectangle(
            width=acc_width,
            height=acc_height,
            color=CELL_INPUT_TEAL,
            fill_color=CELL_INPUT_TEAL,
            fill_opacity=0.3,
            stroke_width=2
        )
        acc_in.move_to([acc_start_x + 2 * acc_spacing, acc_y, 0])
        accumulators['input'] = acc_in

        label_in = Text("Input", color=CELL_INPUT_TEAL, font_size=20)
        label_in.next_to(acc_in, DOWN, buff=0.15)
        acc_labels['input'] = label_in

        # Fade in all accumulators
        self.play(
            FadeIn(acc_ig),
            FadeIn(label_ig),
            FadeIn(acc_og),
            FadeIn(label_og),
            FadeIn(acc_in),
            FadeIn(label_in),
            run_time=1.5
        )
        self.wait(0.5)

        # ====================================================================
        # Animate backprop flow: right to left (t -> t-1 -> t-2)
        # Gates (y_in, y_out) flow first simultaneously, then g(x)
        # ====================================================================
        accumulator_growth = {
            'input': 0,
            'input_gate': 0,
            'output_gate': 0
        }

        for cell_idx in range(2, -1, -1):  # t, t-1, t-2
            cell = cells[cell_idx]

            # Step 1: Create y_in and y_out orbs simultaneously (gates first)
            gate_orbs = []
            gate_targets = []
            gate_types = []

            for comp_idx, (comp_name, color, acc_type) in enumerate(components[:2]):  # y_in, y_out only
                label = component_labels[cell_idx][comp_idx]

                # Create orb at component position
                orb = Circle(
                    radius=0.12,
                    color=color,
                    fill_color=color,
                    fill_opacity=0.9,
                    stroke_width=2
                )
                orb.move_to(label.get_center())
                gate_orbs.append(orb)
                gate_types.append(acc_type)

                # Target position on respective accumulator
                target_acc = accumulators[acc_type]
                target_x = target_acc.get_center()[0]
                target_y = acc_y
                gate_targets.append([target_x, target_y, 0])

            # Fade in gate orbs simultaneously
            self.play(*[FadeIn(orb) for orb in gate_orbs], run_time=0.4)

            # Animate gate orbs flowing to accumulators simultaneously
            self.play(*[
                ApplyMethod(orb.move_to, target)
                for orb, target in zip(gate_orbs, gate_targets)
            ], run_time=0.8)

            # Grow gate accumulators simultaneously
            for i, acc_type in enumerate(gate_types):
                accumulator_growth[acc_type] += 1
                old_acc = accumulators[acc_type]
                new_width = acc_width + accumulator_growth[acc_type] * 0.2

                new_acc = Rectangle(
                    width=new_width,
                    height=acc_height,
                    color=old_acc.get_color(),
                    fill_color=old_acc.get_fill_color(),
                    fill_opacity=0.3,
                    stroke_width=2
                )
                new_acc.move_to(old_acc.get_center())

                accumulators[acc_type] = new_acc

            self.play(
                Transform(acc_ig, accumulators['input_gate']),
                Transform(acc_og, accumulators['output_gate']),
                *[FadeOut(orb) for orb in gate_orbs],
                run_time=0.4
            )

            self.wait(0.2)

            # Step 2: Create g(x) orb (after gates)
            g_label = component_labels[cell_idx][2]  # g is index 2
            g_orb = Circle(
                radius=0.12,
                color=CELL_INPUT_TEAL,
                fill_color=CELL_INPUT_TEAL,
                fill_opacity=0.9,
                stroke_width=2
            )
            g_orb.move_to(g_label.get_center())

            # Target position on Input accumulator
            target_x = acc_in.get_center()[0]
            target_y = acc_y

            self.play(FadeIn(g_orb), run_time=0.3)

            # Animate g(x) orb flowing to Input accumulator
            self.play(
                ApplyMethod(g_orb.move_to, [target_x, target_y, 0]),
                run_time=0.8
            )

            # Grow Input accumulator
            accumulator_growth['input'] += 1
            new_width_in = acc_width + accumulator_growth['input'] * 0.2

            new_acc_in = Rectangle(
                width=new_width_in,
                height=acc_height,
                color=CELL_INPUT_TEAL,
                fill_color=CELL_INPUT_TEAL,
                fill_opacity=0.3,
                stroke_width=2
            )
            new_acc_in.move_to(acc_in.get_center())

            self.play(
                Transform(acc_in, new_acc_in),
                FadeOut(g_orb),
                run_time=0.3
            )

            # Update reference for next iteration
            acc_in = new_acc_in
            accumulators['input'] = acc_in

            self.wait(0.3)

        self.wait(0.5)

        # ====================================================================
        # Takeaway message
        # ====================================================================
        takeaway = Text(
            "Gradients accumulate separately for each component",
            color=TEXT_WHITE,
            font_size=24
        )
        takeaway.to_edge(DOWN, buff=0.3)

        self.play(FadeIn(takeaway, shift=UP * 0.3), run_time=1)
        self.wait(2)

        # Optional: Interactive mode for development
        if os.getenv("MANIM_DEV"):
            self.embed()
