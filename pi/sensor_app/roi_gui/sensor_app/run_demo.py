"""End-to-end demo: replays the test_set frames through the full pipeline
(registration -> OCR -> reading -> voting -> storage -> stub cloud push) and
prints a summary. Uses FolderDriver -- no real hardware involved."""
import logging
import tempfile
from pathlib import Path

from core.nodes import NodeStore
from core.ocr import OCR_MODE, get_default_reader
from drivers.folder import FolderDriver
from pipeline import run_once
from storage.cloud import StubCloudSink
from storage.csv_export import export_daily
from storage.db import fetch_latest_per_label, fetch_minmax, init_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main():
    print(f"OCR_MODE = {OCR_MODE}")

    store = NodeStore("config/nodes.json")
    ocr = get_default_reader()

    work_dir = Path(tempfile.mkdtemp(prefix="sensor_app_demo_"))
    db_path = work_dir / "demo.db"
    db = init_db(db_path)
    cloud = StubCloudSink(work_dir / "cloud_push.jsonl")

    driver = FolderDriver({
        node.node_id: node.driver_spec["frames_dir"] for node in store
    })

    n_cycles = 10
    for cycle in range(n_cycles):
        for node in store:
            results = run_once(node, driver, ocr, db, cloud)
            for r in results:
                print(
                    f"[cycle {cycle}] {node.node_id}/{r['label']}: "
                    f"value={r['value']!r} unit={r['unit']} valid={r['valid']} "
                    f"health_ok={r['health_ok']} raw={r.get('raw_text')!r}"
                )

    print("\n--- latest readings ---")
    for row in fetch_latest_per_label(db):
        print(row)

    print("\n--- min/max ---")
    for row in fetch_minmax(db):
        print(row)

    out_csv = export_daily(db_path, work_dir / "exports", __import__("time").strftime("%Y-%m-%d", __import__("time").gmtime()))
    print(f"\nCSV export written to {out_csv}")
    print(f"Stub cloud JSONL at {work_dir / 'cloud_push.jsonl'}")
    print(f"Demo DB at {db_path}")


if __name__ == "__main__":
    main()
