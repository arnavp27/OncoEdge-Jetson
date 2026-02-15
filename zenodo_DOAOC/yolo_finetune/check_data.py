"""
Dataset Integrity Checker — Validates images, labels, and polygons
"""
import os
from PIL import Image
from pathlib import Path

DATASET = r"E:\4thYear\EMbedding_AI\Project\datasets\oral_cancer_seg"

print("=" * 60)
print("  DATASET INTEGRITY CHECK")
print("=" * 60)

stats = {
    "ok": 0,
    "corrupt_img": [],
    "bad_label": [],
    "missing_label": [],
    "empty_label": 0,
    "total_polygons": 0,
}

for split in ["train", "val"]:
    img_dir = os.path.join(DATASET, "images", split)
    lbl_dir = os.path.join(DATASET, "labels", split)

    imgs = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    lbls = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]

    print(f"\n[{split.upper()}]")
    print(f"  Images: {len(imgs)}")
    print(f"  Labels: {len(lbls)}")

    for img_name in imgs:
        img_path = os.path.join(img_dir, img_name)
        stem = Path(img_name).stem
        lbl_path = os.path.join(lbl_dir, stem + ".txt")

        # 1. Verify image is not corrupt
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception as e:
            stats["corrupt_img"].append(f"{split}/{img_name}: {e}")
            continue

        # 2. Check label file exists
        if not os.path.exists(lbl_path):
            stats["missing_label"].append(f"{split}/{img_name}")
            continue

        # 3. Validate label content
        with open(lbl_path, "r") as f:
            content = f.read().strip()

        if not content:
            # Empty file = Healthy / negative sample (valid)
            stats["empty_label"] += 1
            stats["ok"] += 1
            continue

        # 4. Check each polygon line
        is_bad = False
        for line_num, line in enumerate(content.split("\n"), 1):
            parts = line.strip().split()

            # Need class_id + at least 3 x,y pairs = 7 values minimum
            if len(parts) < 7:
                stats["bad_label"].append(
                    f"{split}/{stem}.txt L{line_num}: only {len(parts)} values (need >=7)"
                )
                is_bad = True
                break

            try:
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                if cls_id not in [0, 1, 2]:
                    stats["bad_label"].append(
                        f"{split}/{stem}.txt L{line_num}: invalid class_id={cls_id}"
                    )
                    is_bad = True
                    break

                if len(coords) % 2 != 0:
                    stats["bad_label"].append(
                        f"{split}/{stem}.txt L{line_num}: odd coord count ({len(coords)})"
                    )
                    is_bad = True
                    break

                out_of_range = [c for c in coords if c < -0.001 or c > 1.001]
                if out_of_range:
                    stats["bad_label"].append(
                        f"{split}/{stem}.txt L{line_num}: coords outside [0,1]"
                    )
                    is_bad = True
                    break

                stats["total_polygons"] += 1

            except ValueError as e:
                stats["bad_label"].append(
                    f"{split}/{stem}.txt L{line_num}: parse error: {e}"
                )
                is_bad = True
                break

        if not is_bad:
            stats["ok"] += 1

# ── REPORT ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"  Valid image-label pairs : {stats['ok']}")
print(f"  Empty labels (Healthy)  : {stats['empty_label']}")
print(f"  Total polygons          : {stats['total_polygons']}")
print(f"  Corrupt images          : {len(stats['corrupt_img'])}")
print(f"  Bad labels              : {len(stats['bad_label'])}")
print(f"  Missing labels          : {len(stats['missing_label'])}")

if stats["corrupt_img"]:
    print("\n  CORRUPT IMAGES:")
    for x in stats["corrupt_img"][:15]:
        print(f"    x {x}")

if stats["bad_label"]:
    print("\n  BAD LABELS:")
    for x in stats["bad_label"][:15]:
        print(f"    x {x}")

if stats["missing_label"]:
    print("\n  MISSING LABELS (first 15):")
    for x in stats["missing_label"][:15]:
        print(f"    x {x}")

if not stats["corrupt_img"] and not stats["bad_label"] and not stats["missing_label"]:
    print("\n  ALL DATA IS CLEAN - Ready for training!")
else:
    print("\n  Issues found. Review the errors above.")
