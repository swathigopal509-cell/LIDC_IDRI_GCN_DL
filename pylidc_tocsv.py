import os
import math
import shutil
import csv
import numpy as np

# --- pylidc compatibility updates ---
for _alias, _builtin in (('int', int), ('float', float), ('bool', bool),
                         ('object', object), ('long', int)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _builtin)

# pylidc unconditionally imports matplotlib.pyplot in Scan.py for its visualize() helpers.  
import sys as _sys, types as _types
if 'matplotlib' not in _sys.modules:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        _mpl     = _types.ModuleType('matplotlib')
        _pyplot  = _types.ModuleType('matplotlib.pyplot')
        _patches = _types.ModuleType('matplotlib.patches')
        _mpl.pyplot = _pyplot
        _sys.modules['matplotlib']         = _mpl
        _sys.modules['matplotlib.pyplot']  = _pyplot
        _sys.modules['matplotlib.patches'] = _patches
# ---------------------------------------------------------------------------

import pydicom
import pylidc as pl
from pylidc.utils import consensus
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cluster import KMeans

OUTPUT_DIR       = r"C:\Users\RainInBlue\OneDrive\Desktop\Wayne\MLG\project\LIDC_dimenssion\output_records_1"
MIN_DIAMETER_MM  = 3.0  # minimum average diameter across radiologists to keep a nodule (mm)
MIN_RADIOLOGISTS = 3    # keep nodules annotated by >= this many readers
CONSENSUS_LEVEL  = 0.5  # pylidc consensus clevel: >=50% agreement per voxel

CHAR_FEATURES = ['malignancy', 'texture', 'spiculation', 'lobulation',
                 'margin', 'sphericity', 'subtlety']

def split_overmerged_cluster(anns):
    """Split a >4-annotation cluster into valid sub-clusters using k-means on centroids."""
    k = math.ceil(len(anns) / 4)
    centroids = np.array([a.centroid for a in anns])
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(centroids)
    return [[a for a, lbl in zip(anns, labels) if lbl == i] for i in range(k)]

def concordant_score(scores):
    """Individual score that minimises total absolute deviation from the group."""
    if not scores:
        return None
    return min(scores, key=lambda s: sum(abs(s - o) for o in scores))

def process_scan(patient_id: str, output_dir: str,
                 min_diameter_mm: float) -> tuple[list[dict], list[str]]:
    """Return (nodule rows, log lines) for a single LIDC scan/patient."""
    logs: list[str] = []
    rows: list[dict] = []

    scan = (pl.query(pl.Scan)
              .filter(pl.Scan.patient_id == patient_id)
              .first())
    if scan is None:
        logs.append(f"  {patient_id}: no scan found in pylidc DB")
        return rows, logs

    try:
        nodule_clusters = scan.cluster_annotations()
    except Exception as exc:
        logs.append(f"  {patient_id}: cluster_annotations failed — {exc}")
        return rows, logs

    if not nodule_clusters:
        return rows, logs

    # Split any over-merged clusters (>4 annotations = two close nodules blended together).
    expanded = []
    for c in nodule_clusters:
        if len(c) > 4:
            sub_clusters = split_overmerged_cluster(c)
            logs.append(f"  {patient_id}: split over-merged cluster ({len(c)} anns) "
                        f"into {len(sub_clusters)} sub-clusters")
            expanded.extend(sub_clusters)
        else:
            expanded.append(c)
    nodule_clusters = expanded

    # Load every DICOM slice for this series (ordered by z).
    try:
        images = scan.load_all_dicom_images(verbose=False)
    except Exception as exc:
        logs.append(f"  {patient_id}: load_all_dicom_images failed — {exc}")
        return rows, logs

    series_uid = scan.series_instance_uid
    study_uid  = scan.study_instance_uid
    px_sp      = float(scan.pixel_spacing)     # pylidc assumes isotropic in-plane
    thickness  = float(scan.slice_thickness)

    logs.append(f"  Patient: {patient_id}  nodules={len(nodule_clusters)}  "
                f"slices={len(images)}  spacing={px_sp}mm  thickness={thickness}mm")

    for nod_idx, anns in enumerate(nodule_clusters):
        n_rad = len(anns)
        if n_rad < MIN_RADIOLOGISTS:
            continue

        # Mean diameter across the radiologists who marked this nodule (mm).
        avg_diam_mm = round(float(np.mean([a.diameter for a in anns])), 2)
        if avg_diam_mm <= min_diameter_mm:
            continue

        # Consensus mask (voxels agreed on by >= CONSENSUS_LEVEL of readers).
        try:
            cmask, cbbox, _ = consensus(anns, clevel=CONSENSUS_LEVEL,
                                        pad=[(0, 0), (0, 0), (0, 0)])
        except Exception as exc:
            logs.append(f"    nodule {nod_idx}: consensus failed — {exc}")
            continue

        if cmask.sum() == 0:
            continue

        row_slice, col_slice, z_slice = cbbox       # (y, x, z) slices in full volume
        z_indices = list(range(z_slice.start, z_slice.stop))
        n_slices  = len(z_indices)

        # HU stats + volume using the consensus mask, slice by slice.
        hu_arrays: list[np.ndarray] = []
        volume_mm3 = 0.0

        for k, z_idx in enumerate(z_indices):
            if z_idx >= len(images):
                continue
            ds        = images[z_idx]
            slope     = float(getattr(ds, 'RescaleSlope',     1))
            intercept = float(getattr(ds, 'RescaleIntercept', 0))
            pixel_arr = ds.pixel_array.astype(np.float32)

            mask2d = cmask[:, :, k]
            if not mask2d.any():
                continue

            full_mask = np.zeros(pixel_arr.shape, dtype=bool)
            full_mask[row_slice, col_slice] = mask2d

            area_mm2   = int(np.sum(full_mask)) * px_sp * px_sp
            volume_mm3 += area_mm2 * thickness

            voxels = pixel_arr[full_mask] * slope + intercept
            if voxels.size > 0:
                hu_arrays.append(voxels)

        log_vol = round(math.log(volume_mm3), 4) if volume_mm3 > 0 else None
        if hu_arrays:
            all_hu  = np.concatenate(hu_arrays)
            hu_mean = round(float(np.mean(all_hu)), 2)
            hu_std  = round(float(np.std(all_hu)),  2)
        else:
            hu_mean = hu_std = None

        # Centroid of the consensus mask (y, x) in image/pixel coordinates.
        yy, xx, _ = np.where(cmask)
        cy = round(float(np.mean(yy)) + row_slice.start, 1)
        cx = round(float(np.mean(xx)) + col_slice.start, 1)

        # Mid-slice DICOM — copy it into OUTPUT_DIR/<patient>/ for convenience.
        mid_k   = n_slices // 2
        mid_idx = z_indices[mid_k]
        mid_ds  = images[mid_idx]

        src_path = getattr(mid_ds, 'filename', None)
        dest_dir = os.path.join(output_dir, patient_id)
        os.makedirs(dest_dir, exist_ok=True)

        if src_path and os.path.exists(src_path):
            filename  = os.path.basename(src_path)
            dest_path = os.path.join(dest_dir, filename)
            if not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)
        else:
            sop = getattr(mid_ds, 'SOPInstanceUID', f'slice_{mid_idx}')
            filename  = f"{sop}.dcm"
            dest_path = os.path.join(dest_dir, filename)
            if not os.path.exists(dest_path):
                pydicom.dcmwrite(dest_path, mid_ds)

        # Z positions of every slice that intersects the nodule.
        z_positions = []
        for zi in z_indices:
            if zi < len(images) and hasattr(images[zi], 'ImagePositionPatient'):
                z_positions.append(round(float(images[zi].ImagePositionPatient[2]), 2))

        # Aggregate radiologist characteristic scores.
        char_vals = {feat: [getattr(a, feat) for a in anns] for feat in CHAR_FEATURES}

        nodule_id = f"{patient_id}_N{nod_idx:02d}"

        row = {
            'PatientID'     : patient_id,
            'Study_UID'     : study_uid,
            'Series_UID'    : series_uid,
            'Nodule_ID'     : nodule_id,
            'Filename'      : filename,
            'Z_Min'         : min(z_positions) if z_positions else None,
            'Z_Max'         : max(z_positions) if z_positions else None,
            'Num_Slices'    : n_slices,
            'Radiologists'  : n_rad,
            'Diameter_mm'   : avg_diam_mm,
            'Centroid_X'    : cx,
            'Centroid_Y'    : cy,
            'Log_Volume_mm3': log_vol,
            'HU_mean'       : hu_mean,
            'HU_std'        : hu_std,
        }

        for feat in CHAR_FEATURES:
            row[feat.capitalize()] = concordant_score(char_vals[feat])

        rows.append(row)
        logs.append(f"    Nodule {nodule_id}: slices={n_slices}  "
                    f"diam={avg_diam_mm}mm  vol={volume_mm3:.1f}mm³  HU_mean={hu_mean}")

    return rows, logs

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Querying scans from pylidc...")
    patient_ids = sorted({s.patient_id for s in pl.query(pl.Scan).all()})
    total = len(patient_ids)
    print(f"  -> {total} patients found in pylidc DB.")

    all_rows: list[dict] = []
    nodule_count = 0

    #parallel Execution of scans
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_scan, pid, OUTPUT_DIR, MIN_DIAMETER_MM): pid
            for pid in patient_ids
        }
        for done, future in enumerate(as_completed(futures), 1):
            pid = futures[future]
            try:
                rows, logs = future.result()
            except Exception as exc:
                print(f"  [{done}/{total}] {pid} — worker raised {exc}")
                continue

            print(f"  [{done}/{total}] {pid}")
            for line in logs:
                print(line)
            all_rows.extend(rows)
            nodule_count += len(rows)

    print(f"\nDone: {total} patients processed, {nodule_count} nodules recorded.")

    if not all_rows:
        print("No qualifying nodules found.")
        return

    headers = list(all_rows[0].keys())

    train_rows = [r for r in all_rows if r.get('Malignancy') != 3]
    test_rows  = [r for r in all_rows if r.get('Malignancy') == 3]

    csv_path  = os.path.join(OUTPUT_DIR, "nodule_records.csv")
    test_path = os.path.join(OUTPUT_DIR, "nodule_records_test.csv")

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(train_rows)

    with open(test_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(test_rows)

    print(f"\nCSV saved:      {csv_path}  ({len(train_rows)} nodules, malignancy != 3)")
    print(f"Test CSV saved: {test_path}  ({len(test_rows)} nodules, malignancy == 3)")

if __name__ == '__main__':
    main()