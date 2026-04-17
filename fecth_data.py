import os
import math
import shutil
import xml.etree.ElementTree as ET
import pydicom
import numpy as np
import openpyxl
from openpyxl.utils import get_column_letter
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageDraw

CHAR_FEATURES  = ['malignancy', 'texture', 'spiculation', 'lobulation', 'margin', 'sphericity', 'subtlety']
MAX_Z_GAP_MM   = 5.0   # max z-gap (mm) between slices of the same nodule
MIN_VOTES      = 2     # min radiologists that must include a pixel for the consensus mask

# ---------------------------------------------------------------------------
def get_xml_data(xml_path):
    """Parses LIDC XML and returns (study_uid, series_uid, z_data).

    z_data: z_pos -> {
        'count'       : int,                   # radiologists who annotated this z
        'centroids'   : [(cx, cy), ...],
        'diameters_px': [d, ...],
        'contour_pts' : [[(x,y), ...], ...],   # one contour list per radiologist
        'nodule_ids'  : [...],
        'malignancy'  : [...], ...              # characteristic values per radiologist
    }
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        def find_t(parent, tag):
            return [e for e in parent.iter() if e.tag.endswith(tag)]

        uid_node = find_t(root, 'SeriesInstanceUid')
        if not uid_node:
            return None, None, {}
        series_uid = uid_node[0].text

        study_node = find_t(root, 'StudyInstanceUid')
        study_uid  = study_node[0].text if study_node else ''

        def empty_z():
            return {'count': 0, 'centroids': [], 'diameters_px': [], 'contour_pts': [],
                    'nodule_ids': [], **{f: [] for f in CHAR_FEATURES}}

        z_data = defaultdict(empty_z)

        for session in find_t(root, 'readingSession'):
            for nodule in find_t(session, 'unblindedReadNodule'):
                nid_nodes = find_t(nodule, 'noduleID')
                nodule_id = nid_nodes[0].text.strip() if nid_nodes and nid_nodes[0].text else ''

                roi_pts = defaultdict(list)
                all_pts = []

                for roi in find_t(nodule, 'roi'):
                    z_node = find_t(roi, 'imageZposition')
                    if not z_node or z_node[0].text is None:
                        continue
                    z  = round(float(z_node[0].text), 2)
                    xs = [float(e.text) for e in find_t(roi, 'xCoord') if e.text is not None]
                    ys = [float(e.text) for e in find_t(roi, 'yCoord') if e.text is not None]
                    pts = list(zip(xs, ys))
                    if pts:
                        roi_pts[z].extend(pts)
                        all_pts.extend(pts)

                if not all_pts:
                    continue

                xs, ys  = zip(*all_pts)
                diam_px = max(max(xs) - min(xs), max(ys) - min(ys))

                char_vals = {}
                for cnode in find_t(nodule, 'characteristics'):
                    for feat in CHAR_FEATURES:
                        nodes = find_t(cnode, feat)
                        if nodes and nodes[0].text:
                            char_vals[feat] = float(nodes[0].text)

                for z, pts in roi_pts.items():
                    cx, cy = mean_centroid(pts)
                    z_data[z]['count']        += 1
                    z_data[z]['centroids'].append((cx, cy))
                    z_data[z]['diameters_px'].append(diam_px)
                    z_data[z]['contour_pts'].append(pts)
                    z_data[z]['nodule_ids'].append(nodule_id)
                    for feat in CHAR_FEATURES:
                        if feat in char_vals:
                            z_data[z][feat].append(char_vals[feat])

        return study_uid, series_uid, dict(z_data)
    except Exception:
        return None, None, {}

# ---------------------------------------------------------------------------
def make_consensus_mask(contour_pts_list, rows, cols):
    """Binary mask where each pixel True if ≥ MIN_VOTES radiologists included it."""
    n_votes = min(MIN_VOTES, len(contour_pts_list))
    vote_map = np.zeros((rows, cols), dtype=np.int16)
    for pts in contour_pts_list:
        if len(pts) < 3:
            continue
        img = Image.new('L', (cols, rows), 0)
        ImageDraw.Draw(img).polygon(pts, fill=1)
        vote_map += np.array(img, dtype=np.int16)
    return vote_map >= n_votes

# ---------------------------------------------------------------------------
def avg_or_none(vals):
    return round(sum(vals) / len(vals), 2) if vals else None

def concordant_score(scores: list) -> float | None:
    """Return the individual score most concordant with the group.

    Picks the score that minimizes total absolute deviation from all others —
    i.e. the radiologist whose reading agrees most with the rest of the panel.
    Ties naturally resolve to the value closest to the group median.
    """
    if not scores:
        return None
    return min(scores, key=lambda s: sum(abs(s - o) for o in scores))

def mean_centroid(pts, decimals=1):
    xs, ys = zip(*pts)
    return round(sum(xs) / len(xs), decimals), round(sum(ys) / len(ys), decimals)

# ---------------------------------------------------------------------------
def process_xml(xml_path: str, series_to_dir: dict, output_dir: str,
                min_diameter_mm: float) -> tuple[list[dict], list[str]]:
    """Process one XML file. Returns (nodule_rows, log_lines)."""
    logs: list[str] = []
    rows: list[dict] = []

    st_uid, s_uid, z_data = get_xml_data(xml_path)
    if not s_uid or not z_data:
        return rows, logs

    dicom_folder = series_to_dir.get(s_uid)
    if dicom_folder is None:
        return rows, logs

    logs.append(f"    Series : {s_uid}")
    logs.append(f"    Folder : {dicom_folder}")

    # Read all DICOMs in the series folder
    z_to_path : dict[float, str] = {}
    p_id      = "Unknown"
    study_uid = st_uid or ''
    row_sp    = col_sp = None
    thickness = None

    for dcm_file in os.listdir(dicom_folder):
        if not dcm_file.endswith(".dcm"):
            continue
        dcm_path = os.path.join(dicom_folder, dcm_file)
        try:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            if not hasattr(ds, 'ImagePositionPatient'):
                continue
            z = round(float(ds.ImagePositionPatient[2]), 2)
            z_to_path[z] = dcm_path
            if p_id == "Unknown" and hasattr(ds, 'PatientID'):
                p_id = str(ds.PatientID)
            if not study_uid and hasattr(ds, 'StudyInstanceUID'):
                study_uid = str(ds.StudyInstanceUID)
            if row_sp is None and hasattr(ds, 'PixelSpacing'):
                row_sp, col_sp = float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])
            if thickness is None and hasattr(ds, 'SliceThickness'):
                thickness = float(ds.SliceThickness)
        except Exception:
            continue

    if not z_to_path:
        return rows, logs

    if row_sp is None:
        row_sp = 1.0
    if col_sp is None:
        col_sp = 1.0

    # Estimate thickness from z-spacing if tag absent
    if thickness is None:
        if len(z_to_path) > 1:
            sorted_zs = sorted(z_to_path.keys())
            thickness = round(
                sum(abs(b - a) for a, b in zip(sorted_zs, sorted_zs[1:])) / (len(sorted_zs) - 1), 4
            )
        else:
            thickness = 1.0

    px_to_mm = (row_sp + col_sp) / 2.0
    logs.append(f"    Patient: {p_id}  slices={len(z_to_path)}"
                f"  spacing={row_sp}x{col_sp}mm  thickness={thickness}mm")

    # Filter qualifying z-positions
    qualified = []
    for z_pos, info in z_data.items():
        if info['count'] < 3:
            continue
        avg_diam_mm = round(
            (sum(info['diameters_px']) / len(info['diameters_px'])) * px_to_mm, 2
        )
        if avg_diam_mm <= min_diameter_mm:
            continue
        if z_pos not in z_to_path:
            continue
        qualified.append((z_pos, info, avg_diam_mm))

    if not qualified:
        logs.append(f"    No qualifying nodule slices — skipping.")
        return rows, logs

    # Group consecutive z-positions into nodules
    qualified.sort(key=lambda x: x[0])
    nodule_groups: list[list] = [[qualified[0]]]
    for item in qualified[1:]:
        if abs(item[0] - nodule_groups[-1][-1][0]) <= MAX_Z_GAP_MM:
            nodule_groups[-1].append(item)
        else:
            nodule_groups.append([item])

    # Compute features per nodule
    for group in nodule_groups:
        n_slices = len(group)
        all_z    = [item[0] for item in group]
        mid_idx  = n_slices // 2
        mid_z, mid_info, avg_diam_mm = group[mid_idx]
        mid_path = z_to_path[mid_z]

        cx, cy = mean_centroid(mid_info['centroids'])

        char_agg : dict[str, list] = defaultdict(list)
        all_nids : list[str]       = []
        for z_pos, info, _ in group:
            for feat in CHAR_FEATURES:
                char_agg[feat].extend(info[feat])
            all_nids.extend(info['nodule_ids'])

        nodule_id = Counter(all_nids).most_common(1)[0][0] if all_nids else ''

        volume_mm3 = 0.0
        hu_arrays  : list[np.ndarray] = []

        for z_pos, info, _ in group:
            src_path = z_to_path.get(z_pos)
            if src_path is None:
                continue
            try:
                ds        = pydicom.dcmread(src_path)
                rows_dim  = int(ds.Rows)
                cols_dim  = int(ds.Columns)
                slope     = float(getattr(ds, 'RescaleSlope',     1))
                intercept = float(getattr(ds, 'RescaleIntercept', 0))
                pixel_arr = ds.pixel_array.astype(np.float32)

                mask      = make_consensus_mask(info['contour_pts'], rows_dim, cols_dim)
                area_mm2  = int(np.sum(mask)) * row_sp * col_sp
                volume_mm3 += area_mm2 * thickness

                voxels = pixel_arr[mask] * slope + intercept
                if voxels.size > 0:
                    hu_arrays.append(voxels)
            except Exception:
                continue

        log_vol = round(math.log(volume_mm3), 4) if volume_mm3 > 0 else None
        if hu_arrays:
            all_hu  = np.concatenate(hu_arrays)
            hu_mean = round(float(np.mean(all_hu)), 2)
            hu_std  = round(float(np.std(all_hu)),  2)
        else:
            hu_mean = hu_std = None

        filename = os.path.basename(mid_path)
        dest_dir  = os.path.join(output_dir, p_id)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        if not os.path.exists(dest_path):
            shutil.copy2(mid_path, dest_path)

        row = {
            'PatientID'      : p_id,
            'Study_UID'      : study_uid,
            'Series_UID'     : s_uid,
            'Nodule_ID'      : nodule_id,
            'Filename'       : filename,
            'Z_Min'          : min(all_z),
            'Z_Max'          : max(all_z),
            'Num_Slices'     : n_slices,
            'Radiologists'   : mid_info['count'],
            'Diameter_mm'    : avg_diam_mm,
            'Centroid_X'     : cx,
            'Centroid_Y'     : cy,
            'Log_Volume_mm3' : log_vol,
            'HU_mean'        : hu_mean,
            'HU_std'         : hu_std,
        }
        mal = char_agg['malignancy']
        row['Malignancy']      = concordant_score(mal)
        row['Malignancy_Mean'] = avg_or_none(mal)
        row['Malignancy_Std']  = round(float(np.std(mal)), 2) if mal else None

        for feat in CHAR_FEATURES:
            if feat == 'malignancy':
                continue
            row[feat.capitalize()] = avg_or_none(char_agg[feat])

        rows.append(row)
        logs.append(f"    Nodule {nodule_id}: {filename} | slices={n_slices}"
                    f" | diam={avg_diam_mm}mm | vol={volume_mm3:.1f}mm³ | HU_mean={hu_mean}")

    return rows, logs

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DICOM_DIR       = r"C:\Users\RainInBlue\Downloads\dataset\TCIA_LIDC-IDRI_20200921\lidc_idri"
XML_DIR         = r"C:\Users\RainInBlue\Downloads\dataset\xml\LIDC-XML-only\tcia-lidc-xml"
OUTPUT_DIR      = r"C:\Users\RainInBlue\OneDrive\Desktop\Wayne\MLG\project\LIDC_dimenssion\output_records_1"
MIN_DIAMETER_MM = 3.0

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # STEP 1 — Build a fast SeriesUID → folder index
    #           Read ONE DICOM per directory to identify the series.
    print("Indexing DICOM directories (one header per folder)...")
    series_to_dir: dict[str, str] = {}   # SeriesUID -> folder path

    for root_dir, _, files in os.walk(DICOM_DIR):
        dcm_files = [f for f in files if f.endswith(".dcm")]
        if not dcm_files:
            continue
        try:
            ds = pydicom.dcmread(os.path.join(root_dir, dcm_files[0]), stop_before_pixels=True)
            if hasattr(ds, 'SeriesInstanceUID'):
                series_to_dir[str(ds.SeriesInstanceUID)] = root_dir
        except Exception:
            continue

    print(f"  -> {len(series_to_dir)} series directories indexed.")

    # STEP 2 — Collect all XML paths, then process them in parallel.
    xml_paths = [
        os.path.join(root, fname)
        for root, _, files in os.walk(XML_DIR)
        for fname in sorted(files) if fname.endswith('.xml')
    ]
    total = len(xml_paths)
    print(f"\nProcessing {total} XMLs in parallel...")

    excel_rows  : list[dict] = []
    nodule_count = 0

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_xml, p, series_to_dir, OUTPUT_DIR, MIN_DIAMETER_MM): p
            for p in xml_paths
        }
        for done, future in enumerate(as_completed(futures), 1):
            xml_path = futures[future]
            rows, logs = future.result()
            print(f"  [{done}/{total}] {os.path.basename(xml_path)}")
            for line in logs:
                print(line)
            excel_rows.extend(rows)
            nodule_count += len(rows)

    print(f"\nDone: {total} XMLs processed, {nodule_count} nodules recorded.")

    # SAVE EXCEL
    if excel_rows:
        xlsx_path = os.path.join(OUTPUT_DIR, "nodule_records.xlsx")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        wb = openpyxl.Workbook()
        ws = wb.active or wb.create_sheet("Nodule Records")
        ws.title = "Nodule Records"

        headers = list(excel_rows[0].keys())
        ws.append(headers)
        for row in excel_rows:
            ws.append([row[h] for h in headers])

        for i, col in enumerate(ws.columns, start=1):
            max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
            ws.column_dimensions[get_column_letter(i)].width = min(max_len + 2, 40)

        wb.save(xlsx_path)
        print(f"\nExcel saved: {xlsx_path}  ({len(excel_rows)} nodules)")
    else:
        print("\nNo qualifying nodules found.")
