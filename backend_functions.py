from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

STATE_FILE = Path(__file__).resolve().parent / "floorplan_state.json"

FT_TO_MM = 304.8
DEFAULT_CEILING_FT = 10.0
DEFAULT_WALL_THICKNESS_MM = 150
DEFAULT_DOOR_HEIGHT_FT = 7.0
DEFAULT_WINDOW_HEIGHT_FT = 4.0
DEFAULT_WINDOW_SILL_FT = 3.0


# -----------------------------
# Core helpers
# -----------------------------
def to_mm(value: float, unit: str) -> int:
    u = (unit or "ft").lower()
    if u in ("mm", "millimeter", "millimeters"):
        return int(value)
    if u in ("m", "meter", "meters", "metre", "metres"):
        return int(value * 1000)
    return int(value * FT_TO_MM)


def norm_unit(unit: str | None) -> str:
    if not unit:
        return "ft"
    u = unit.lower()
    if u in ("feet", "foot"):
        return "ft"
    return u


def empty_structure() -> dict[str, Any]:
    return {
        "units": "mm",
        "space": None,
        "exterior_walls": [],
        "openings": [],
        "columns": [],
        "inner_walls": [],
        "beams": [],
        "services": {"points": []},
        "meta": {"door_counter": 0, "window_counter": 0},
    }


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return empty_structure()
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return empty_structure()


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def wall_length(wall: dict[str, Any]) -> int:
    x1, y1 = wall["start"]
    x2, y2 = wall["end"]
    return int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)


def get_wall(state: dict[str, Any], wall_id: str) -> dict[str, Any] | None:
    w = wall_id.upper()
    for wall in state["exterior_walls"]:
        if wall["id"] == w:
            return wall
    return None


def center_offset(wall: dict[str, Any], elem_width_mm: int) -> int:
    length = wall_length(wall)
    return max(0, int((length - elem_width_mm) / 2))


def next_opening_id(state: dict[str, Any], opening_type: str) -> str:
    if opening_type == "door":
        state["meta"]["door_counter"] += 1
        return f"D{state['meta']['door_counter']}"
    state["meta"]["window_counter"] += 1
    return f"WIN{state['meta']['window_counter']}"


def opening_segment(
    state: dict[str, Any], opening: dict[str, Any]
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    wall = get_wall(state, opening.get("wall_id", ""))
    if not wall:
        return None
    x1, y1 = wall["start"]
    x2, y2 = wall["end"]
    dx = x2 - x1
    dy = y2 - y1
    length = (dx**2 + dy**2) ** 0.5
    if length <= 0:
        return None
    ux = dx / length
    uy = dy / length
    off = float(opening.get("offset_from_start", 0))
    width = float(opening.get("width", 0))
    sx = x1 + ux * off
    sy = y1 + uy * off
    ex = sx + ux * width
    ey = sy + uy * width
    return (sx, sy), (ex, ey)


# -----------------------------
# Extracted functions
# -----------------------------
def reset() -> dict[str, Any]:
    state = empty_structure()
    save_state(state)
    return {
        "status": "success",
        "message": "State reset",
        "state_file": str(STATE_FILE),
    }


def get_structure() -> dict[str, Any]:
    return load_state()


def create_space(length: float, width: float, unit: str = "ft") -> dict[str, Any]:
    state = load_state()
    length_mm = to_mm(length, unit)
    width_mm = to_mm(width, unit)
    height_mm = to_mm(DEFAULT_CEILING_FT, "ft")

    state["space"] = {
        "name": "Room",
        "height": height_mm,
        "footprint": {"min_x": 0, "min_y": 0, "max_x": length_mm, "max_y": width_mm},
    }

    t = DEFAULT_WALL_THICKNESS_MM
    state["exterior_walls"] = [
        {
            "id": "W1",
            "start": [0, 0],
            "end": [0, width_mm],
            "thickness": t,
            "inward_normal": [1, 0],
        },
        {
            "id": "W2",
            "start": [0, 0],
            "end": [length_mm, 0],
            "thickness": t,
            "inward_normal": [0, 1],
        },
        {
            "id": "W3",
            "start": [length_mm, 0],
            "end": [length_mm, width_mm],
            "thickness": t,
            "inward_normal": [-1, 0],
        },
        {
            "id": "W4",
            "start": [0, width_mm],
            "end": [length_mm, width_mm],
            "thickness": t,
            "inward_normal": [0, -1],
        },
    ]

    state["openings"] = []
    state["columns"] = []
    state["inner_walls"] = []
    state["beams"] = []
    state["services"] = {"points": []}
    state["meta"] = {"door_counter": 0, "window_counter": 0}

    save_state(state)
    return {
        "status": "success",
        "message": f"Space created: {length} x {width} {unit}",
        "space": state["space"],
    }


def add_opening(
    wall_id: str,
    width: float,
    unit: str = "ft",
    opening_type: str = "door",
    height: float | None = None,
    offset: float | None = None,
) -> dict[str, Any]:
    state = load_state()
    if not state.get("space"):
        return {"status": "error", "message": "Create space first"}

    wall = get_wall(state, wall_id)
    if not wall:
        return {"status": "error", "message": f"Invalid wall_id: {wall_id}"}

    width_mm = to_mm(width, unit)
    if height is None:
        height = (
            DEFAULT_DOOR_HEIGHT_FT
            if opening_type == "door"
            else DEFAULT_WINDOW_HEIGHT_FT
        )
    height_mm = to_mm(height, "ft")

    if offset is None:
        off_mm = center_offset(wall, width_mm)
    else:
        off_mm = to_mm(offset, unit)

    max_len = wall_length(wall)
    if off_mm < 0 or off_mm + width_mm > max_len:
        return {"status": "error", "message": "Opening out of wall bounds"}

    oid = next_opening_id(state, opening_type)
    item = {
        "id": oid,
        "type": "door" if opening_type == "door" else "window",
        "wall_id": wall["id"],
        "offset_from_start": off_mm,
        "width": width_mm,
        "height": height_mm,
    }
    if opening_type == "window":
        item["sill_height"] = to_mm(DEFAULT_WINDOW_SILL_FT, "ft")

    state["openings"].append(item)
    save_state(state)
    return {"status": "success", "message": f"{item['type']} added", "opening": item}


def add_door(
    wall_id: str,
    width: float,
    unit: str = "ft",
    height: float | None = None,
    offset: float | None = None,
) -> dict[str, Any]:
    return add_opening(
        wall_id, width, unit=unit, opening_type="door", height=height, offset=offset
    )


def add_window(
    wall_id: str,
    width: float,
    unit: str = "ft",
    height: float | None = None,
    offset: float | None = None,
) -> dict[str, Any]:
    return add_opening(
        wall_id, width, unit=unit, opening_type="window", height=height, offset=offset
    )


def delete_opening(opening_id: str) -> dict[str, Any]:
    state = load_state()
    target = opening_id.upper()
    before = len(state["openings"])
    state["openings"] = [
        o for o in state["openings"] if str(o.get("id", "")).upper() != target
    ]
    if len(state["openings"]) == before:
        return {"status": "error", "message": f"Opening not found: {opening_id}"}
    save_state(state)
    return {"status": "success", "message": f"Deleted {opening_id}"}


def show() -> dict[str, Any]:
    return get_structure()


def points() -> dict[str, Any]:
    state = load_state()
    out: dict[str, Any] = {
        "space": state.get("space", {}),
        "walls": [],
        "openings": [],
    }

    for wall in state.get("exterior_walls", []):
        out["walls"].append(
            {
                "id": wall.get("id"),
                "start": wall.get("start"),
                "end": wall.get("end"),
            }
        )

    for opening in state.get("openings", []):
        seg = opening_segment(state, opening)
        if not seg:
            continue
        start, end = seg
        out["openings"].append(
            {
                "id": opening.get("id"),
                "type": opening.get("type"),
                "wall_id": opening.get("wall_id"),
                "offset_from_start": opening.get("offset_from_start"),
                "width": opening.get("width"),
                "start": [round(start[0], 2), round(start[1], 2)],
                "end": [round(end[0], 2), round(end[1], 2)],
            }
        )

    return out


def render_svg(out_path: str = "floorplan.svg") -> dict[str, Any]:
    state = load_state()
    print(state)
    p = Path(out_path).expanduser().resolve()

    if not state.get("space"):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="400">'
            '<rect width="100%" height="100%" fill="white"/>'
            '<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" '
            'font-family="sans-serif" font-size="20" fill="#666">Create space first</text>'
            "</svg>"
        )
        p.write_text(svg)
        return {"status": "success", "message": "SVG written", "path": str(p)}

    fp = state["space"]["footprint"]
    min_x, min_y = fp["min_x"], fp["min_y"]
    max_x, max_y = fp["max_x"], fp["max_y"]
    pad = 200
    vb_x = min_x - pad
    vb_y = -(max_y + pad)
    vb_w = (max_x - min_x) + (2 * pad)
    vb_h = (max_y - min_y) + (2 * pad)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb_x} {vb_y} {vb_w} {vb_h}">'
    )
    parts.append(
        '<rect x="-100000" y="-100000" width="200000" height="200000" fill="white"/>'
    )
    parts.append('<g transform="scale(1,-1)">')

    for wall in state.get("exterior_walls", []):
        x1, y1 = wall["start"]
        x2, y2 = wall["end"]
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            'stroke="#111" stroke-width="30" stroke-linecap="square"/>'
        )

    for o in state.get("openings", []):
        seg = opening_segment(state, o)
        if not seg:
            continue
        (sx, sy), (ex, ey) = seg
        color = "#0b84f3" if o.get("type") == "window" else "#cc4b37"
        parts.append(
            f'<line x1="{sx}" y1="{sy}" x2="{ex}" y2="{ey}" '
            f'stroke="{color}" stroke-width="36" stroke-linecap="butt"/>'
        )
        mx = (sx + ex) / 2
        my = (sy + ey) / 2
        parts.append(
            f'<text x="{mx}" y="{my + 80}" fill="#111" font-size="120" text-anchor="middle" '
            f'font-family="sans-serif" transform="scale(1,-1) translate(0,{-2 * (my + 80)})">{o.get("id", "")}</text>'
        )

    parts.append("</g>")
    parts.append("</svg>")
    p.write_text("\n".join(parts))
    return {"status": "success", "message": "SVG written", "path": str(p)}


def render_png(out_path: str = "floorplan.png") -> dict[str, Any]:
    svg_tmp = Path(out_path).expanduser().resolve().with_suffix(".tmp.svg")
    svg_result = render_svg(str(svg_tmp))
    if svg_result.get("status") != "success":
        return svg_result

    out = Path(out_path).expanduser().resolve()

    try:
        import cairosvg  # type: ignore

        cairosvg.svg2png(url=str(svg_tmp), write_to=str(out))
        try:
            svg_tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return {
            "status": "success",
            "message": "PNG written",
            "path": str(out),
            "engine": "cairosvg",
        }
    except Exception:
        pass

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        state = load_state()
        if not state.get("space"):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "Create space first", ha="center", va="center")
            ax.axis("off")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            try:
                svg_tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return {
                "status": "success",
                "message": "PNG written",
                "path": str(out),
                "engine": "matplotlib",
            }

        fig, ax = plt.subplots(figsize=(8, 6))
        for wall in state.get("exterior_walls", []):
            x1, y1 = wall["start"]
            x2, y2 = wall["end"]
            ax.plot([x1, x2], [y1, y2], color="#222", linewidth=10)

        for o in state.get("openings", []):
            seg = opening_segment(state, o)
            if not seg:
                continue
            (sx, sy), (ex, ey) = seg
            color = "#0b84f3" if o.get("type") == "window" else "#cc4b37"
            ax.plot([sx, ex], [sy, ey], color=color, linewidth=12)
            ax.text((sx + ex) / 2, (sy + ey) / 2, str(o.get("id", "")), fontsize=9)

        fp = state["space"]["footprint"]
        ax.set_xlim(fp["min_x"] - 200, fp["max_x"] + 200)
        ax.set_ylim(fp["min_y"] - 200, fp["max_y"] + 200)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        try:
            svg_tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return {
            "status": "success",
            "message": "PNG written",
            "path": str(out),
            "engine": "matplotlib",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "PNG render failed. Install cairosvg or matplotlib.",
            "detail": str(e),
        }


def run_prompt(text: str) -> dict[str, Any]:
    t = text.strip()
    tl = t.lower()

    if tl in ("reset", "clear", "start over"):
        return {"parsed": "reset", "result": reset()}
    if tl in ("show", "show state", "current state"):
        return {"parsed": "show", "result": show()}

    m = re.search(
        r"(?:create|make)?\s*(?:space|room).*?(\d+(?:\.\d+)?)\s*(ft|feet|foot|m|mm)?\s*(?:x|by)\s*(\d+(?:\.\d+)?)\s*(ft|feet|foot|m|mm)?",
        tl,
    )
    if m:
        l = float(m.group(1))
        unit1 = norm_unit(m.group(2))
        w = float(m.group(3))
        unit2 = norm_unit(m.group(4))
        unit = unit1 if unit1 != "ft" or not m.group(4) else unit2
        return {"parsed": "create-space", "result": create_space(l, w, unit)}

    m = re.search(
        r"(?:add|put)\s+(door|window).*?(?:on\s+)?(w[1-4]).*?(\d+(?:\.\d+)?)\s*(ft|feet|foot|m|mm)?",
        tl,
    )
    if m:
        kind = m.group(1)
        wall = m.group(2).upper()
        width = float(m.group(3))
        unit = norm_unit(m.group(4))
        if kind == "door":
            return {"parsed": "add-door", "result": add_door(wall, width, unit)}
        return {"parsed": "add-window", "result": add_window(wall, width, unit)}

    m = re.search(r"(?:delete|remove)\s+(d\d+|win\d+)", tl)
    if m:
        oid = m.group(1).upper()
        return {"parsed": "delete-opening", "result": delete_opening(oid)}

    return {
        "status": "error",
        "message": "Prompt not recognized",
        "examples": [
            "create space 10 by 8 ft",
            "add door on W2 3 ft",
            "add window on W4 4 ft",
            "delete D1",
            "show",
            "reset",
        ],
    }


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone floor-plan script (no server dependency)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("reset")
    sub.add_parser("show")
    sub.add_parser("points")

    sp = sub.add_parser("create-space")
    sp.add_argument("--length", type=float, required=True)
    sp.add_argument("--width", type=float, required=True)
    sp.add_argument("--unit", default="ft")

    d = sub.add_parser("add-door")
    d.add_argument("--wall", required=True)
    d.add_argument("--width", type=float, required=True)
    d.add_argument("--unit", default="ft")
    d.add_argument("--height", type=float)
    d.add_argument("--offset", type=float)

    w = sub.add_parser("add-window")
    w.add_argument("--wall", required=True)
    w.add_argument("--width", type=float, required=True)
    w.add_argument("--unit", default="ft")
    w.add_argument("--height", type=float)
    w.add_argument("--offset", type=float)

    do = sub.add_parser("delete-opening")
    do.add_argument("--id", required=True)

    rsvg = sub.add_parser("render-svg")
    rsvg.add_argument("--out", default="floorplan.svg")

    rpng = sub.add_parser("render-png")
    rpng.add_argument("--out", default="floorplan.png")

    pr = sub.add_parser("prompt")
    pr.add_argument("--text", required=True)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "reset":
        out = reset()
    elif args.cmd == "show":
        out = show()
    elif args.cmd == "points":
        out = points()
    elif args.cmd == "create-space":
        out = create_space(args.length, args.width, args.unit)
    elif args.cmd == "add-door":
        out = add_door(args.wall, args.width, args.unit, args.height, args.offset)
    elif args.cmd == "add-window":
        out = add_window(args.wall, args.width, args.unit, args.height, args.offset)
    elif args.cmd == "delete-opening":
        out = delete_opening(args.id)
    elif args.cmd == "render-svg":
        out = render_svg(args.out)
    elif args.cmd == "render-png":
        out = render_png(args.out)
    elif args.cmd == "prompt":
        out = run_prompt(args.text)
    else:
        out = {"status": "error", "message": "Unknown command"}

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
