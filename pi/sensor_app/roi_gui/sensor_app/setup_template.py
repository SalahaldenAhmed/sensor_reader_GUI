"""Build a locked node template from a reference frame + manually picked boxes,
and write/update config/nodes.json. No VLM/YOLO — manual ROIs + heuristics only.

Usage (as a library):
    from setup_template import build_node_template, write_node_to_config
    node_cfg = build_node_template(
        node_id="htc2_cam",
        reference_image_path="assets/test_set/frames/frame_001.jpg",
        driver_spec={"type": "folder", "frames_dir": "assets/test_set/frames"},
        manual_boxes=[
            {"label": "IN_TEMP", "x": 760, "y": 360, "w": 180, "h": 90, "sample_text": "25.3C"},
            ...
        ],
    )
    write_node_to_config(node_cfg, "config/nodes.json")
"""
import json
from pathlib import Path
from typing import List, Optional

from core.reading import parse_reference


def _infer_unit_type(label: str, sample_text: Optional[str]) -> "tuple[str, Optional[str]]":
    """Heuristic unit/type inference from the field label and/or a sample OCR
    string captured at setup time. '%' -> humidity, ':' -> clock, else temp/C."""
    text = sample_text or ""
    label_l = label.lower()
    if "%" in text or "hum" in label_l:
        return "humidity", "%RH"
    if ":" in text or "time" in label_l or "clock" in label_l:
        return "clock", None
    return "temp", "C"


def build_box_template(label: str, x: float, y: float, w: float, h: float,
                        sample_text: Optional[str] = None,
                        range_min: Optional[float] = None,
                        range_max: Optional[float] = None) -> dict:
    box_type, unit = _infer_unit_type(label, sample_text)

    if box_type == "clock":
        return {
            "label": label, "x": x, "y": y, "w": w, "h": h,
            "type": "clock", "unit": None, "decimals": 0, "int_digits": 0,
            "range_min": None, "range_max": None, "ref_value": None, "last_good": None,
        }

    ref_value, int_digits, decimals = parse_reference(sample_text or "")
    if int_digits is None:
        int_digits, decimals = 2, (0 if box_type == "humidity" else 1)

    if box_type == "humidity" and (range_min is None or range_max is None):
        range_min, range_max = 0.0, 100.0

    return {
        "label": label, "x": x, "y": y, "w": w, "h": h,
        "type": box_type, "unit": unit, "decimals": decimals, "int_digits": int_digits,
        "range_min": range_min, "range_max": range_max,
        "ref_value": ref_value, "last_good": ref_value,
    }


def build_node_template(node_id: str, reference_image_path: str, driver_spec: dict,
                         manual_boxes: List[dict]) -> dict:
    boxes = []
    for mb in manual_boxes:
        boxes.append(build_box_template(
            label=mb["label"], x=mb["x"], y=mb["y"], w=mb["w"], h=mb["h"],
            sample_text=mb.get("sample_text"),
            range_min=mb.get("range_min"), range_max=mb.get("range_max"),
        ))
    return {
        "node_id": node_id,
        "driver_spec": driver_spec,
        "reference_image": reference_image_path,
        "boxes": boxes,
    }


def write_node_to_config(node_cfg: dict, config_path: str = "config/nodes.json") -> None:
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
    else:
        data = {"nodes": []}

    data["nodes"] = [n for n in data["nodes"] if n["node_id"] != node_cfg["node_id"]]
    data["nodes"].append(node_cfg)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
