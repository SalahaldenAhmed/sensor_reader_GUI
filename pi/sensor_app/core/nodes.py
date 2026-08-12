"""Node/Box data model + NodeStore that loads config/nodes.json and holds
runtime state (last_good, voters, health) per node."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from core.reading import TemporalVoter
from core.registration import build_reference


@dataclass
class Box:
    label: str
    x: float
    y: float
    w: float
    h: float
    type: str
    unit: Optional[str]
    decimals: int
    int_digits: int
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    ref_value: Optional[float] = None
    last_good: Optional[float] = None

    def to_field_dict(self) -> dict:
        """Shape consumed by core.reading.read_anchored."""
        return {
            "type": self.type,
            "unit": self.unit,
            "decimals": self.decimals,
            "int_digits": self.int_digits,
            "range_min": self.range_min,
            "range_max": self.range_max,
            "ref_value": self.ref_value,
            "last_good": self.last_good,
        }

    def to_box_dict(self) -> dict:
        """Shape consumed by core.registration.register_and_warp."""
        return {
            "label": self.label, "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "type": self.type, "unit": self.unit, "decimals": self.decimals,
            "int_digits": self.int_digits, "range_min": self.range_min,
            "range_max": self.range_max, "ref_value": self.ref_value,
            "last_good": self.last_good,
        }


@dataclass
class Node:
    node_id: str
    driver_spec: dict
    reference: dict
    boxes: List[Box]
    voters: Dict[str, TemporalVoter] = field(default_factory=dict)
    minmax_trackers: Dict[str, "MinMaxTracker"] = field(default_factory=dict)
    last_health: Optional[dict] = None

    def box(self, label: str) -> Box:
        for b in self.boxes:
            if b.label == label:
                return b
        raise KeyError(f"No box with label {label!r} on node {self.node_id!r}")

    def voter_for(self, label: str) -> TemporalVoter:
        if label not in self.voters:
            is_clock = self.box(label).type in ("clock", "time")
            self.voters[label] = TemporalVoter(is_clock=is_clock)
        return self.voters[label]

    def minmax_for(self, conn, label: str) -> "MinMaxTracker":
        from core.minmax import MinMaxTracker
        if label not in self.minmax_trackers:
            self.minmax_trackers[label] = MinMaxTracker.load_or_create(conn, self.node_id, label)
        return self.minmax_trackers[label]


def _infer_unit_type(label: str, hint: Optional[str]) -> None:
    pass  # placeholder kept out; inference lives in setup_template.py


class NodeStore:
    """Loads config/nodes.json into Node objects, building each node's ORB
    reference from its stored reference image."""

    def __init__(self, config_path: str = "config/nodes.json", base_dir: Optional[str] = None):
        self.config_path = Path(config_path)
        self.base_dir = Path(base_dir) if base_dir else self.config_path.resolve().parent.parent
        self.nodes: Dict[str, Node] = {}
        self._load()

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (self.base_dir / path)

    def _load(self) -> None:
        with open(self.config_path) as f:
            data = json.load(f)

        for node_cfg in data["nodes"]:
            ref_image_path = self._resolve(node_cfg["reference_image"])
            ref_frame = cv2.imread(str(ref_image_path))
            if ref_frame is None:
                raise FileNotFoundError(f"Could not read reference image {ref_image_path}")
            reference = build_reference(ref_frame)

            boxes = [Box(**b) for b in node_cfg["boxes"]]

            node = Node(
                node_id=node_cfg["node_id"],
                driver_spec=node_cfg["driver_spec"],
                reference=reference,
                boxes=boxes,
            )
            self.nodes[node.node_id] = node

    def __iter__(self):
        return iter(self.nodes.values())

    def get(self, node_id: str) -> Node:
        return self.nodes[node_id]
