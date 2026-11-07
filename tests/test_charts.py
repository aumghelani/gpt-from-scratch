import itertools
import math
import re

import pytest

from gpt_scratch.report.svg import delta_bar_svg, line_chart_svg, nice_ticks


def test_nice_ticks_cover_range_with_round_steps():
    ticks = nice_ticks(1.5, 2.6, 5)
    assert ticks[0] <= 1.5 and ticks[-1] >= 2.6
    steps = {round(b - a, 9) for a, b in itertools.pairwise(ticks)}
    assert len(steps) == 1 and next(iter(steps)) in (0.1, 0.2, 0.25, 0.5)


def test_nice_ticks_degenerate_range():
    assert len(nice_ticks(3.0, 3.0, 5)) >= 2


def test_line_chart_has_one_polyline_per_series_and_legend():
    svg = line_chart_svg({"a": [(0, 3), (1, 2), (2, 1)], "b": [(0, 2.5), (2, 0.5)]}, title="T",
                         x_label="step", y_label="loss")
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert len(re.findall(r"<polyline", svg)) == 2
    assert ">a<" in svg and ">b<" in svg and ">T<" in svg


def test_labels_are_xml_escaped():
    svg = line_chart_svg({"<script>x</script>": [(0, 1), (1, 2)]}, title="a & b", x_label="x", y_label="y")
    assert "<script>" not in svg and "&lt;script&gt;" in svg and "a &amp; b" in svg


def test_single_point_series_still_renders():
    assert "<circle" in line_chart_svg({"a": [(0, 1.0)]}, title="t", x_label="x", y_label="y")


@pytest.mark.parametrize("series", [{}, {"a": []}])
def test_empty_input_rejected(series):
    with pytest.raises(ValueError):
        line_chart_svg(series, title="t", x_label="x", y_label="y")


def test_non_finite_values_rejected():
    with pytest.raises(ValueError):
        line_chart_svg({"a": [(0, math.nan), (1, 1)]}, title="t", x_label="x", y_label="y")


def test_theme_changes_background():
    light = line_chart_svg({"a": [(0, 1), (1, 2)]}, "t", "x", "y", theme="light")
    dark = line_chart_svg({"a": [(0, 1), (1, 2)]}, "t", "x", "y", theme="dark")
    assert light != dark


def test_delta_bars_color_improvements_and_show_values():
    svg = delta_bar_svg([("+rope", -0.073), ("worse", 0.05)], title="Δ", x_label="val loss vs baseline")
    assert svg.count("<rect") >= 3  # background + 2 bars
    assert "-0.073" in svg and "+0.050" in svg


def test_delta_bars_empty_rejected():
    with pytest.raises(ValueError):
        delta_bar_svg([], title="t", x_label="x")


def test_longest_negative_bar_leaves_room_for_its_value_label():
    """Regression: the label of the longest bar used to overprint the row label."""
    svg = delta_bar_svg([("+swiglu", -0.092), ("+rope", -0.073)], title="t", x_label="x")
    bar_xs = [float(m) for m in re.findall(r'<rect x="([\d.]+)" y="\d+" width', svg)]
    label_right_edge = 150 - 12  # row labels are right-aligned at (left - 12)
    value_label_width = 56  # "-0.092" at 13px, with a gap
    assert min(bar_xs) >= label_right_edge + value_label_width
