"""Reading interpretation: anchored digit decoding + temporal voting.

The three functions below (parse_reference, parse_device_range, read_anchored)
are PORTED VERBATIM from the existing setup/runtime logic. Do not change their
semantics — only the surrounding plumbing (TemporalVoter) is new.
"""
import re
from collections import deque
from typing import Optional


def parse_reference(*texts):
    for t in texts:
        if not t:
            continue
        m = re.search(r"-?\d+\.\d+", t.replace("O", "0").replace("o", "0"))
        if m:
            num = m.group(0)
            intp, _, frac = num.lstrip("-").partition(".")
            return float(num), len(intp), len(frac)
    for t in texts:
        if not t:
            continue
        m = re.search(r"-?\d+", t.replace("O", "0"))
        if m:
            return float(m.group(0)), len(m.group(0).lstrip("-")), 0
    return None, None, None


def parse_device_range(box_texts):
    for t in box_texts:
        has_unit = ("℃" in t) or ("°" in t) or ("C" in t)
        looks_range = ("~" in t) or (" to " in t.lower()) or t.count("-") >= 1
        if has_unit and looks_range:
            nums = re.findall(r"-?\d+", t)
            if len(nums) >= 2:
                a, b = int(nums[0]), int(nums[1])
                return min(a, b), max(a, b), t
    return None, None, None


def read_anchored(raw_paddle, field):
    dec = field["decimals"]
    total = (field["int_digits"] or 2) + dec
    if field["type"] in ("clock", "time"):
        digits = re.sub(r"[^0-9]", "", raw_paddle)
        if len(digits) == 3:
            return f"{digits[0]}:{digits[1:]}", True, ""
        if len(digits) >= 4:
            return f"{digits[:2]}:{digits[2:4]}", True, ""
        return raw_paddle, False, "clock: too few digits"
    digits = re.sub(r"[^0-9]", "", raw_paddle)
    neg = raw_paddle.strip().startswith("-")
    if total <= 0 or len(digits) < total:
        return None, False, f"too few digits ({digits!r})"
    cands = []
    for i in range(len(digits) - total + 1):
        w = digits[i:i + total]
        v = f"{w[:-dec]}.{w[-dec:]}" if dec else w
        cands.append(float(("-" if neg else "") + v))
    lo, hi = field.get("range_min"), field.get("range_max")
    inrange = [c for c in cands if lo <= c <= hi] if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) else cands
    pool = inrange or cands
    anchor = field.get("last_good") or field.get("ref_value")
    best = min(pool, key=lambda c: abs(c - anchor)) if anchor is not None else pool[0]
    valid = (lo <= best <= hi) if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) else True
    if valid:
        field["last_good"] = best
    return f"{best:g}", valid, "" if valid else "out of range"


class TemporalVoter:
    """Publishes a value only once the most recent N accepted readings agree.

    Keeps a ring buffer of up to K accepted readings. commit(value) pushes the
    new reading and returns (stable_value, is_stable):
      - is_stable=True and stable_value=value when the last N readings agree.
      - is_stable=False and stable_value=<previous stable value, possibly None>
        when they don't yet agree (flapping input holds the previous stable
        reading rather than publishing noise).
    Clock-type values ("HH:MM" strings) are compared by exact match; numeric
    values are compared within `epsilon`.
    """

    def __init__(self, k: int = 5, n_agree: int = 3, epsilon: float = 1e-6, is_clock: bool = False):
        if n_agree > k:
            raise ValueError("n_agree cannot exceed k")
        self.k = k
        self.n_agree = n_agree
        self.epsilon = epsilon
        self.is_clock = is_clock
        self._buf = deque(maxlen=k)
        self._stable_value = None

    def _agree(self, a, b) -> bool:
        if self.is_clock:
            return a == b
        try:
            return abs(float(a) - float(b)) <= self.epsilon
        except (TypeError, ValueError):
            return a == b

    def commit(self, value) -> "tuple[Optional[object], bool]":
        self._buf.append(value)
        recent = list(self._buf)[-self.n_agree:]
        if len(recent) == self.n_agree and all(self._agree(v, recent[0]) for v in recent):
            self._stable_value = recent[0]
            return self._stable_value, True
        return self._stable_value, False

    def reset(self) -> None:
        self._buf.clear()
        self._stable_value = None
