#!/usr/bin/env python
"""T1 — Convierte CrowdPose (14 kp) a esqueleto COCO-17 en formato YOLO-pose.

Mapeo semántico documentado (plan H3, roadmap r8 #11):
- Los 12 puntos de cuerpo van a su slot COCO equivalente.
- `head_top` y `neck` se DESCARTAN (semánticamente ≠ nose/eyes/ears).
- Slots faciales COCO 0-4 quedan v=0 (no supervisados en CrowdPose).
El orden de keypoints de CrowdPose se LEE del JSON (categories[0].keypoints),
nunca de memoria. Genera splits train/val separados + un YAML de dataset mixto
(CrowdPose-train + subset seeded de COCO-pose train) + validación visual.

Uso:
    python scripts/convert_crowdpose.py            # convierte + valida
    python scripts/convert_crowdpose.py --visual-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CP = REPO / "data" / "crowdpose"
OUT = REPO / "data" / "crowdpose_yolo"
COCO_DIR = REPO / "data" / "coco-pose"

# COCO-17 canónico (idéntico a harmocap.schema.KEYPOINT_ORDER)
COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle"]
# flip_idx COCO-17 auditado contra el orden de arriba (pack §B.4)
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# nombre CrowdPose → slot COCO-17 (los no listados se descartan)
SEMANTIC_MAP = {
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}
DISCARDED = {"head", "head_top", "top_head", "neck"}

COCO_SUBSET_N = 20_000
COCO_SUBSET_SEED = 0


def audit_flip_idx() -> None:
    """flip_idx debe mapear cada punto a su espejo L/R según COCO17 real."""
    for i, name in enumerate(COCO17):
        j = FLIP_IDX[i]
        if name.startswith("left_"):
            assert COCO17[j] == name.replace("left_", "right_"), (i, name)
        elif name.startswith("right_"):
            assert COCO17[j] == name.replace("right_", "left_"), (i, name)
        else:
            assert j == i, (i, name)
    print("[convert] flip_idx auditado contra el orden COCO-17 real ✓")


def convert_split(ann_path: Path, split: str, manifest: dict) -> None:
    data = json.loads(ann_path.read_text())
    kp_names = data["categories"][0]["keypoints"]      # orden REAL del JSON
    print(f"[convert] {split}: orden de keypoints del JSON = {kp_names}")

    # mapa índice CrowdPose → slot COCO (validando nombres contra SEMANTIC_MAP)
    idx_map: dict[int, int] = {}
    for i, name in enumerate(kp_names):
        n = name.lower()
        if n in SEMANTIC_MAP:
            idx_map[i] = SEMANTIC_MAP[n]
        elif n in DISCARDED:
            continue
        else:
            raise SystemExit(f"keypoint desconocido en CrowdPose: {name!r} — "
                             "revisar SEMANTIC_MAP antes de continuar")
    if len(idx_map) != 12:
        raise SystemExit(f"esperaba mapear 12 puntos de cuerpo, mapeé {len(idx_map)}")

    images = {im["id"]: im for im in data["images"]}
    by_img = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd"):
            continue
        by_img[ann["image_id"]].append(ann)

    img_dir = OUT / "images" / split
    lbl_dir = OUT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_persons = 0
    src_images = CP / "images"
    for img_id, anns in by_img.items():
        im = images[img_id]
        w, h = im["width"], im["height"]
        src = src_images / im["file_name"]
        if not src.exists():
            continue
        # symlink (no copiar 2.5 GB)
        dst = img_dir / im["file_name"]
        if not dst.exists():
            dst.symlink_to(src.resolve())
        lines = []
        for ann in anns:
            bx, by_, bw, bh = ann["bbox"]
            if bw <= 1 or bh <= 1:
                continue
            cx, cy = (bx + bw / 2) / w, (by_ + bh / 2) / h
            kps = ann["keypoints"]
            out_kp = [[0.0, 0.0, 0] for _ in range(17)]
            for ci, slot in idx_map.items():
                x, y, v = kps[ci * 3: ci * 3 + 3]
                if v > 0:
                    out_kp[slot] = [x / w, y / h, min(int(v), 2)]
            flat = " ".join(f"{x:.6f} {y:.6f} {v}" for x, y, v in out_kp)
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f} {flat}")
            n_persons += 1
        (lbl_dir / (Path(im["file_name"]).stem + ".txt")).write_text(
            "\n".join(lines) + "\n")

    manifest[split] = {
        "images": len(by_img), "persons": n_persons,
        "source_json": ann_path.name,
        "source_sha256": hashlib.sha256(ann_path.read_bytes()).hexdigest()[:16],
        "keypoint_order_source": kp_names,
    }
    print(f"[convert] {split}: {len(by_img)} imágenes, {n_persons} personas")


def build_mixed_yaml(manifest: dict) -> None:
    """YAML de dataset mixto: CrowdPose train + subset seeded de COCO train."""
    coco_train_list = COCO_DIR / "train2017.txt"
    assert coco_train_list.exists(), "COCO-pose no descargado aún"
    all_coco = coco_train_list.read_text().strip().splitlines()
    rng = random.Random(COCO_SUBSET_SEED)
    subset = sorted(rng.sample(all_coco, min(COCO_SUBSET_N, len(all_coco))))
    subset_file = REPO / "data" / "coco_subset_train.txt"
    # rutas relativas al datasets_dir (data/)
    subset_file.write_text("\n".join(f"./coco-pose/{p.lstrip('./')}" for p in subset) + "\n")
    manifest["coco_subset"] = {"n": len(subset), "seed": COCO_SUBSET_SEED,
                               "from_total": len(all_coco)}

    mixed = f"""# Dataset mixto H3: CrowdPose(train, convertido a COCO-17) + subset COCO-pose
# Generado por scripts/convert_crowdpose.py — manifest en data/crowdpose_yolo/manifest.json
path: {REPO / 'data'}
train:
  - crowdpose_yolo/images/train
  - coco_subset_train.txt
val: crowdpose_yolo/images/val    # eval principal (oclusión); COCO val se evalúa aparte
kpt_shape: [17, 3]
flip_idx: {FLIP_IDX}
names:
  0: person
"""
    (REPO / "configs" / "dataset_mixed_ft1.yaml").write_text(mixed)
    print(f"[convert] dataset mixto: configs/dataset_mixed_ft1.yaml "
          f"(+{len(subset)} imgs COCO seed={COCO_SUBSET_SEED})")


def visual_check(n: int = 24) -> None:
    """Renderiza n imágenes convertidas con el esqueleto para inspección."""
    import cv2
    pairs = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12),
             (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
    out_dir = OUT / "visual_check"
    out_dir.mkdir(exist_ok=True)
    lbls = sorted((OUT / "labels" / "train").glob("*.txt"))[:n]
    for lp in lbls:
        ip = OUT / "images" / "train" / (lp.stem + ".jpg")
        if not ip.exists():
            continue
        img = cv2.imread(str(ip))
        h, w = img.shape[:2]
        for line in lp.read_text().strip().splitlines():
            f = [float(x) for x in line.split()]
            kp = [(f[5 + i * 3] * w, f[6 + i * 3] * h, f[7 + i * 3])
                  for i in range(17)]
            for a, b in pairs:
                if kp[a][2] > 0 and kp[b][2] > 0:
                    cv2.line(img, (int(kp[a][0]), int(kp[a][1])),
                             (int(kp[b][0]), int(kp[b][1])), (0, 255, 0), 2)
            for i, (x, y, v) in enumerate(kp):
                if v > 0:
                    col = (255, 0, 0) if "left" in COCO17[i] else (0, 0, 255)
                    cv2.circle(img, (int(x), int(y)), 4, col, -1)
        cv2.imwrite(str(out_dir / lp.stem) + "_check.jpg", img)
    print(f"[convert] validación visual: {out_dir} "
          f"(izquierda=AZUL, derecha=ROJO — revisar que no estén cruzados)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual-only", action="store_true")
    args = ap.parse_args()
    audit_flip_idx()
    if not args.visual_only:
        manifest: dict = {}
        ann = CP / "annotations"
        # los archivos publicados: crowdpose_{train,val,test}.json (a veces en subcarpeta)
        cands = {p.name: p for p in ann.rglob("*.json")}
        # nombres EXACTOS: "trainval" contiene "train" y "val" — no usar substring
        train_json = cands.get("crowdpose_train.json")
        val_json = cands.get("crowdpose_val.json")
        assert train_json and val_json, f"JSONs no encontrados en {ann}: {list(cands)}"
        convert_split(train_json, "train", manifest)
        convert_split(val_json, "val", manifest)
        build_mixed_yaml(manifest)
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    visual_check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
