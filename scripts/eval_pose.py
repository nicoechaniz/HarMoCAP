#!/usr/bin/env python
"""T2/T4 — Eval reproducible de un checkpoint de pose en los dos benchmarks.

PROTOCOLO FIJO (idéntico para baseline y candidato — comparabilidad):
- imgsz=640, batch=16, device cuda:0, conf/IoU/NMS = defaults de model.val()
- (a) CrowdPose-val convertido a COCO-17 (métrica objetivo: multitud/oclusión)
- (b) COCO-pose val2017 (métrica de regresión: dominio general + cara)
Salida: JSON con pose mAP50-95 / mAP50 por benchmark → reports/<run_id>/.

Uso:
    python scripts/eval_pose.py --checkpoint yolo26m-pose.pt --tag baseline
    python scripts/eval_pose.py --checkpoint runs/.../best.pt --tag ft1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent

# Protocolo pineado (validate-hyperparams): NO cambiar entre corridas
PROTOCOL = dict(imgsz=640, batch=16, device="cuda:0", verbose=False, plots=False)

# YAML del benchmark CrowdPose-val convertido (val del dataset mixto)
CROWDPOSE_YAML = REPO / "configs" / "dataset_mixed_ft1.yaml"
COCO_YAML = "coco-pose.yaml"          # resuelto por ultralytics (datasets_dir)


def run_eval(checkpoint: str, tag: str) -> dict:
    ckpt = Path(checkpoint)
    out: dict = {
        "tag": tag,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": (hashlib.sha256(ckpt.read_bytes()).hexdigest()
                              if ckpt.exists() else "resolved-by-ultralytics"),
        "protocol": {**PROTOCOL, "conf_iou_nms": "defaults de model.val()"},
        "benchmarks": {},
    }
    for name, data in (("crowdpose_val_coco17", str(CROWDPOSE_YAML)),
                       ("coco_pose_val2017", COCO_YAML)):
        model = YOLO(checkpoint)          # instancia fresca por benchmark
        t0 = time.time()
        m = model.val(data=data, **PROTOCOL)
        out["benchmarks"][name] = {
            "pose_map50_95": round(float(m.pose.map), 4),
            "pose_map50": round(float(m.pose.map50), 4),
            "box_map50_95": round(float(m.box.map), 4),
            "eval_s": round(time.time() - t0, 1),
        }
        print(f"[eval:{tag}] {name}: pose mAP50-95={m.pose.map:.4f} "
              f"mAP50={m.pose.map50:.4f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    out = run_eval(args.checkpoint, args.tag)
    run_file = REPO / "reports" / "CURRENT_RUN"
    run_id = run_file.read_text().strip().split("=", 1)[1] if run_file.exists() \
        else "manual"
    dest = REPO / "reports" / run_id / f"eval_{args.tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, allow_nan=False))
    print(f"[eval:{args.tag}] → {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
