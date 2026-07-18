#!/usr/bin/env python
"""M2/M4 — corre el pipeline de tiempo real (plan M2).

Fuente: webcam (índice) o archivo de video. Emite OSC según el contrato v1 y
reporta al final las métricas GO/NO-GO (latencia software p50/p95/p99, jitter
|Δsent−Δcaptured|, drops) que se versionan en reports/<run_id>/.

Uso:
    python scripts/run_realtime.py --source 0 --seconds 60
    python scripts/run_realtime.py --source video.mp4 --record outputs/sessions/s1.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from harmocap.pipeline import HarmocapPipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="0",
                    help="índice de webcam o ruta de video")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--warmup", type=float, default=10.0,
                    help="warmup excluido de métricas (plan r3 #12)")
    ap.add_argument("--record", default=None, help="grabar sesión a .jsonl")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--show", action="store_true",
                    help="ventana con esqueletos + selección de foco: teclas "
                         "1-8 = slot, 0/a = auto, q/ESC = salir")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    dests = [(args.host, args.port)] if args.host and args.port else None
    pipe = HarmocapPipeline(REPO, source=source, record_to=args.record,
                            osc_destinations=dests)
    pipe.camera.start()
    print(f"[run] backend: {pipe.backend.info()}")
    print(f"[run] captura: {pipe.camera.profile()}")
    print(f"[run] stream_id={pipe.stream_id} contract_id={pipe.contract_id}")

    show = args.show
    if show:
        import cv2
    t0 = time.monotonic()
    warmup_done = False
    try:
        while time.monotonic() - t0 < args.warmup + args.seconds:
            if not warmup_done and time.monotonic() - t0 >= args.warmup:
                pipe.metrics["lat_sw_ms"].clear()   # descartar warmup
                pipe.metrics["jitter_ms"].clear()
                warmup_done = True
            if not pipe.step():
                print("[run] fuente agotada")
                break
            if show and getattr(pipe, "last_frame_img", None) is not None:
                img = pipe.last_frame_img.copy()
                h = img.shape[0]
                for p in pipe.last_persons:
                    if not p.present:
                        continue
                    col = (0, 220, 255) if p.focused else (0, 180, 0)
                    for k in p.keypoints:
                        if k.state != 2:   # invalid no se dibuja
                            cv2.circle(img, (int(k.x * h), int(k.y * h)), 3, col, -1)
                    xs = [k.x * h for k in p.keypoints if k.state != 2]
                    ys = [k.y * h for k in p.keypoints if k.state != 2]
                    if xs:
                        cv2.putText(img, f"slot {p.slot_id}" +
                                    (" *FOCO*" if p.focused else ""),
                                    (int(min(xs)), max(14, int(min(ys)) - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                cv2.putText(img, f"foco: {pipe.slots.focus_mode} "
                            f"(slot {pipe.slots.focused_slot})  [1-8]=slot 0/a=auto q=salir",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1)
                cv2.imshow("HarMoCAP", img)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord("0"), ord("a")):
                    pipe.slots.select_auto()
                elif ord("1") <= key <= ord("8"):
                    pipe.slots.select_focus(key - ord("1"))
    except KeyboardInterrupt:
        print("\n[run] interrumpido")
    finally:
        if show:
            cv2.destroyAllWindows()
        report = pipe.report()
        pipe.close()

    print(json.dumps(report, indent=2, default=str))
    run_file = REPO / "reports" / "CURRENT_RUN"
    if run_file.exists():
        run_id = run_file.read_text().strip().split("=", 1)[1]
        out = REPO / "reports" / run_id / "realtime_metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"[run] métricas → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
