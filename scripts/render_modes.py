#!/usr/bin/env python
"""Render comparativo de MODOS (H4): grupo vs masa sobre el mismo video.

Por cada video genera dos mp4:
  <stem>__modo_grupo.mp4  imgsz 640 + BoT-SORT+ReID + reasociación de slots:
                          bbox por slot (color estable, ★ = foco automático)
  <stem>__modo_masa.mp4   imgsz 1280 + ByteTrack: TODAS las detecciones crudas
                          + HUD con los agregados del contrato 1.2 (crowd_count,
                          crowd_qom, density, dispersion) y flecha de flow.

Uso: python scripts/render_modes.py video1.mp4 [...] [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from harmocap.crowd import CrowdAggregator  # noqa: E402
from harmocap.identity import SlotManager  # noqa: E402
from harmocap.perception import PoseBackend  # noqa: E402

COLORS = [(66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53),
          (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36)]

ENGINE = str(REPO / "outputs/harmocap-m-pose-ft2.engine")
FALLBACK = "harmocap-m-pose-ft2.pt"


def _writer(video: Path, dst: Path):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    return cap, vw, fps, w, h


def render_group(video: Path, dst: Path) -> float:
    ident = yaml.safe_load((REPO / "configs/identity.yaml").read_text())
    rq = dict(ident.get("reacquisition", {}), enabled=True)
    backend = PoseBackend(realtime_checkpoint=ENGINE, fallback_checkpoint=FALLBACK,
                          imgsz=640, conf=0.05, max_det=300,
                          tracker=str(REPO / "configs/tracker_group.yaml"))
    slots = SlotManager(
        max_slots=8,
        occlusion_grace_ms=ident["slot"]["occlusion_grace_ms"],
        release_timeout_ms=ident["slot"]["release_timeout_ms"],
        acquire_rule=ident["slot"]["acquire_rule"],
        tombstone_repeat_frames=ident["slot"]["tombstone_repeat_frames"],
        reacquisition=rq)
    cap, vw, fps, w, h = _writer(video, dst)
    t_us, frames, t0 = 0, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        t_us += int(1e6 / fps)
        dets, _raw, _speed, _wh = backend.track_frame(frame)
        events = slots.update(dets, t_us, aspect=w / h if h else 16 / 9)
        focused = slots.focused_slot
        for ev in events:
            if ev.detection is None:
                continue
            cx, cy, bw, bh = ev.detection.bbox_xywhn
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            color = COLORS[ev.slot_id % 8]
            thick = 4 if ev.slot_id == focused else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            tag = f"S{ev.slot_id}" + ("*" if ev.rebound else "") \
                + (" ★" if ev.slot_id == focused else "")
            cv2.putText(frame, tag, (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, "modo GRUPO (640, botsort+reid+reasoc)",
                    (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        vw.write(frame)
    cap.release(); vw.release()
    return frames / (time.time() - t0)


def render_crowd(video: Path, dst: Path) -> float:
    backend = PoseBackend(realtime_checkpoint=ENGINE, fallback_checkpoint=FALLBACK,
                          imgsz=1280, conf=0.05, max_det=300,
                          tracker="bytetrack.yaml")
    agg = CrowdAggregator()
    cap, vw, fps, w, h = _writer(video, dst)
    aspect = w / h if h else 16 / 9
    t_us, frames, t0 = 0, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        t_us += int(1e6 / fps)
        _dets, raw, _speed, _wh = backend.track_frame(frame)
        crowd = agg.update(raw, t_us, aspect=aspect)
        for (cx, cy, bw, bh) in raw:
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (52, 168, 83), 1)
        # HUD contrato 1.2
        hud = [f"crowd_count {crowd['crowd_count']}",
               f"crowd_qom   {crowd['crowd_qom']:.2f}",
               f"density     {crowd['density']:.2f}",
               f"dispersion  {crowd['dispersion']:.2f}"]
        for i, line in enumerate(hud):
            cv2.putText(frame, line, (12, 26 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(frame, line, (12, 26 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # flecha de flow desde el centroide (coords iso → px)
        if crowd["crowd_count"] > 0:
            px = int(crowd["centroid_x"] / aspect * w)
            py = int(crowd["centroid_y"] * h)
            fx = int(crowd["flow_x"] * 0.15 * w)
            fy = int(crowd["flow_y"] * 0.15 * h)
            cv2.arrowedLine(frame, (px, py), (px + fx, py + fy),
                            (251, 188, 5), 3, tipLength=0.3)
        cv2.putText(frame, "modo MASA (1280, bytetrack, detecciones crudas)",
                    (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        vw.write(frame)
    cap.release(); vw.release()
    return frames / (time.time() - t0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path,
                    default=REPO / "Biblioteca/videos_people_dancing_modos")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for v in args.videos:
        for name, fn in (("grupo", render_group), ("masa", render_crowd)):
            dst = args.out / f"{v.stem}__modo_{name}.mp4"
            print(f"[modes] {v.name} / {name} …", flush=True)
            pf = fn(v, dst)
            print(f"[modes]   → {dst.name} ({pf:.1f} fps proceso)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
