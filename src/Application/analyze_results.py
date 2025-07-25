from typing import Dict, Any, Tuple, List

from src.Common.system_io import save_csv

DEBUG = False


def save_results(results: Dict[int, Dict[str, Any]] | None,
                 file_name: str):
    save_csv(results,
             file_name,
             ['related_holes', 'arm_lengths', 'arm_endpoints', 'arm_contours', 'core_contour', 'body_contours'],
             ['core_center', 'core_bounding_box']
             )

def summarise_arms(analyzed: Dict[int, Dict[str, Any]]):
    summery = {
        'file': '',
        'arm_lengths': 'None',
        'total_bodies_count': len(analyzed),
        'core_area': 0,
        'body_area': 0,
        'arms_area': 0,
        'arms/core': 'NaN',
        'core/arms': 'NaN',
        'num_arms': 0,
        'average_arms_width': 'NaN',
        'max_arm_length': 0,
        'avg_arm_length': "NaN",
        'sum_arm_length': 0,
        'tile_row': 'None',
        'tile_col': 'None',
        'x': 'None',
        'y': 'None',
        'core_center': 'None',
        'core_contour': 'None',
        'core_bounding_box': 'None',
        'related_holes': 'None',
        'core_w': 'None',
        'core_h': 'None',
        'arm_contours': 'None',
        'body_contours': 'None',
        'name': 'sum',
    }

    def is_number(value):
        """Checks if a value is a number (int or float). Handles None and empty strings."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return False
        try:
            float(value)  # Try converting to float; if it fails, it's not a number
            return True
        except (TypeError, ValueError):
            return False

    total_arms_count = 0
    total_arms_total_width = 0
    for _, data in analyzed.items():

        summery['core_area'] += data.get('core_area', 0)
        summery['body_area'] += data.get('body_area', 0)
        summery['arms_area'] += data.get('arms_area', 0)
        summery['num_arms'] += data.get('num_arms', 0)
        summery['sum_arm_length'] += data.get('sum_arm_length', 0)

        if data.get('max_arm_length', 0) > summery['max_arm_length']:
            summery['max_arm_length'] = data.get('max_arm_length', 0)

        arms_count = data.get('num_arms')

        arms_avg_width = data.get('average_arms_width')
        if arms_count > 0 and is_number(arms_avg_width):
            total_arms_total_width += arms_count * arms_avg_width
            total_arms_count += arms_count

    summery['arms/core'] = "NaN" if summery['core_area'] == 0 else summery['arms_area'] / summery['core_area']

    summery['core/arms'] = "NaN" if summery['arms_area'] == 0 else summery['core_area'] / summery['arms_area']

    summery[
        'average_arms_width'] = "not calculated or arms count zero " if total_arms_count == 0 else total_arms_total_width / total_arms_count

    summery['avg_arm_length'] = "NaN" if summery['num_arms'] == 0 else summery['sum_arm_length'] / summery['num_arms']

    return summery

def remove_overlapping_cells(cells: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Given a list of cell results from overlapping image tiles, remove duplicate detections.
    Each cell's absolute bounding box is computed by adding its tile offset (x, y) to its
    relative bounding box (core_bounding_box). If two bounding boxes overlap, only the cell
    with the larger core_area is retained.

    Parameters:
        cells: List of dictionaries containing cell data with keys including:
               'core_bounding_box' (tuple: (x, y, w, h)), 'x', 'y', and 'core_area'.

    Returns:
        A dictionary mapping new indices to the cell data dictionaries for the non-overlapping cells.
    """

    def get_absolute_bbox(cell: Dict[str, Any]) -> Tuple[int, int, int, int]:
        # core_bounding_box is (cbb_x, cbb_y, cbb_w, cbb_h)
        cbb_x, cbb_y, cbb_w, cbb_h = cell['core_bounding_box']
        abs_x = cell['x'] + cbb_x
        abs_y = cell['y'] + cbb_y
        return (abs_x, abs_y, cbb_w, cbb_h)

    def boxes_overlap(cell1: Dict[str, Any], cell2: Dict[str, Any]) -> bool:
        x1, y1, w1, h1 = get_absolute_bbox(cell1)
        x2, y2, w2, h2 = get_absolute_bbox(cell2)

        # Compute bottom-right coordinates
        r1_x2, r1_y2 = x1 + w1, y1 + h1
        r2_x2, r2_y2 = x2 + w2, y2 + h2

        # Check for overlap: if one rectangle is on the left or above the other, they don't overlap.
        if x1 >= r2_x2 or x2 >= r1_x2:
            return False
        if y1 >= r2_y2 or y2 >= r1_y2:
            return False
        return True

    n = len(cells)
    # Boolean list to mark cells that will be kept
    keep = [True] * n

    # Compare each pair of cells
    for i in range(n):
        if not keep[i]:
            continue  # Skip cells already marked for removal
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            if cells[i]['tile_row'] ==  cells[j]['tile_row'] \
                    and cells[i]['tile_col'] ==  cells[j]['tile_col']:
                continue
            if boxes_overlap(cells[i], cells[j]):
                if DEBUG:
                    print(f"cells overlapping {cells[i]['name']} : {cells[j]['name']}")

                if cells[i]['core_area'] >= cells[j]['core_area']:
                    keep[j] = False

                else:
                    keep[i] = False
                    break  # No need to compare cell i with others if it's removed

    # Build the final dictionary of cells to return, re-indexed.
    filtered_cells = {}
    new_idx = 0
    for idx, cell in enumerate(cells):
        if keep[idx]:
            filtered_cells[new_idx] = cell
            new_idx += 1
    print(f"Overlapping results: {len(filtered_cells)} / {n}")
    return filtered_cells
