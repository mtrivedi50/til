# datanim — Design Spec

**Date:** 2026-04-29
**Status:** Approved

## Overview

datanim is a Python library for creating Manim-style animations of Polars data operations. Given a chain of data transformations, it renders a video (MP4 or GIF) that visually explains each step — showing rows matching in a join, collapsing in a groupby, or fading out in a filter.

**Primary audience:** Engineers and data practitioners explaining pipelines to teammates, with secondary use in educational content and presentations.

---

## Architecture

Five layers, each with a single responsibility:

```
User API  →  Operation Pipeline  →  Animation Planner  →  SVG Renderer  →  FFmpeg Encoder
```

1. **User API** — fluent chain that collects ops and triggers rendering
2. **Operation Pipeline** — each chained call appends an `Op` object storing its input/output DataFrames
3. **Animation Planner** — converts each `Op` into an ordered sequence of keyframes (highlight, fade, slide, collapse)
4. **SVG Renderer** — draws each keyframe as SVG, rasterizes to PNG via `cairosvg`
5. **FFmpeg Encoder** — stitches the PNG sequence into MP4 or GIF

---

## Project Structure

```
datanim/
├── __init__.py          # exposes animate()
├── api.py               # AnimationBuilder — the fluent chain
├── ops/
│   ├── base.py          # Op base class
│   ├── join.py          # JoinOp
│   ├── filter.py        # FilterOp
│   └── groupby.py       # GroupByOp
├── planner.py           # Op → list of keyframes
├── renderer.py          # keyframes → SVG → PNG via cairosvg
└── encoder.py           # PNG sequence → MP4/GIF via FFmpeg
```

---

## API

```python
import polars as pl
from datanim import animate

animate(df_left) \
    .join(df_right, on="id", how="inner") \
    .filter(pl.col("age") > 25) \
    .groupby("dept").agg(pl.sum("salary")) \
    .render("output.mp4", fps=30)
```

- `animate(df)` — accepts a single starting Polars DataFrame; returns an `AnimationBuilder`
- `.join(other, on, how)` — records a `JoinOp` with `other` as the right DataFrame; `how` defaults to `"inner"`
- `.filter(expr)` — records a `FilterOp` with a Polars expression
- `.groupby(by)` — returns a `GroupByBuilder` intermediate; does not record an op yet
- `.agg(expr)` — called on `GroupByBuilder`; records a `GroupByOp` and returns the `AnimationBuilder`
- `.render(path, fps=30, width=1280, height=720, format="mp4")` — executes the pipeline and writes the output file

---

## Operation → Animation Mapping

### Join
1. Render left and right tables side by side
2. Highlight rows with matching keys (green)
3. Non-matching rows fade out (red → transparent)
4. Matching rows slide together into a merged table

### Filter
1. Render the table with the filter expression shown as a label
2. Rows passing the condition highlight green
3. Rows failing the condition turn red and fade out
4. Remaining rows slide up to close the gaps

### GroupBy + Aggregate
1. Color-code rows by group value
2. Rows in the same group slide together into a cluster
3. Cluster collapses into a single summary row showing the aggregated value

---

## Rendering

- Each keyframe is a full SVG document representing one frame of the animation
- Intermediate frames between keyframes are linearly interpolated (opacity, y-position)
- `cairosvg` rasterizes SVGs to PNG at the target resolution
- FFmpeg encodes the PNG sequence: `ffmpeg -framerate {fps} -i frame_%04d.png -c:v libx264 output.mp4`
- Dark theme by default (Catppuccin Mocha palette) to match Manim's aesthetic

---

## Dependencies

| Package | Purpose |
|---|---|
| `polars` | DataFrame engine |
| `cairosvg` | SVG → PNG rasterization |
| `ffmpeg` (system) | PNG sequence → MP4/GIF |

---

## Out of Scope (v1)

- Derived/formula columns (stretch goal for v2)
- Pandas support
- Interactive web output
- Audio narration
- Jupyter inline display
