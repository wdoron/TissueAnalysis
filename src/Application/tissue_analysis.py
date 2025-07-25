import os
import time
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple

from skimage.measure import shannon_entropy

from src.Application.analyze_results import save_results
from src.Common.config import MIN_CORE_AREA, MAX_CORE_AREA, CORE_CIRCLE_RADIUS_MASK, MAX_DAB_AREA, MIN_DAB_AREA, \
    MIN_AVG_CORE_DAB_COLOR, MAX_ARM_LENGTH_FOR_VALID, MIN_DAB_STD_FOR_VALID_IMAGE, MIN_CORE_BODY_DIFF, MIN_ARMS_LENGTH, \
    CORE_SECONDARY_THRESHOLD_PERCENT, CORE_SECONDARY_THRESHOLD, CORE_SECONDARY_THRESHOLD_KERNEL_SIZE, USE_FFT
from src.Common.histogram_utils import (
    histogram_avg_std,
    create_histogram_as_image,
    get_histogram_percentage,
    get_hist_values,
)
from src.Common.image_processing import (
    skeletonize_image,
    find_contours,
    normalize_image,
    multiple_in_mask,
    sample_widths_in_mask_distance,
    create_circles_mask, fft_bandpass_filter,
)
from src.Common.json_info import JsonInfo
from src.Common.stain_separation import StainSeparation
from src.Common.system_io import display_images, save_data_as_json
from src.Common.utils import debug_image, format_time

# === Debug Flags ===
DEBUG_CORES = False
DEBUG_BODY_PHASE1 = False
DEBUG_BODY_PHASE2 = False
DEBUG_BODY_AND_CORES = False

# === Feature Flags ===
SAMPLE_WIDTH_FLAG = False

# === Core Analysis Thresholds and Morphological Constants ===
FFT_LOW_FREQ = 10
FFT_HIGH_FREQ = 300

CORE_GAUSSIAN_KERNEL = (21, 21)
CORE_GAUSSIAN_SIGMA = 43

CORE_PRIMARY_THRESHOLD = 65
CORE_PRIMARY_THRESHOLD_PERCENT = 0.99
CORE_HEM_THRESHOLD = 160
CORE_HEM_THRESHOLD_PERCENT = 0.9
CORE_HEM_THRESHOLD_KERNEL_SIZE = 3

#**** under config
# CORE_SECONDARY_THRESHOLD = 101
# CORE_SECONDARY_THRESHOLD_PERCENT = 0.3
# CORE_SECONDARY_THRESHOLD_KERNEL_SIZE = 2#0

CORE_THRESH_KERNEL_SIZE = 9
CORE_THRESH_ITERATIONS = 1
CORE_THRESH_KERNEL_OPEN_SIZE = 0

# === Body Analysis Thresholds and Morphological Constants ===
BODY_GAUSSIAN_KERNEL = (31, 31)
BODY_GAUSSIAN_SIGMA = 50

BODY_SKELETON_THRESHOLD = 70
BODY_SKELETON_THRESHOLD_PERCENT = 0.95
BODY_SKELETON_THRESH_KERNEL_SIZE = 5
BODY_SKELETONIZE_IMAGE_KERNEL_SIZE = 3

BODY_ENHANCEMENT_THRESHOLD = 60
BODY_ENHANCEMENT_THRESHOLD_PERCENT = 0.86
BODY_ENHANCEMENT_THRESH_KERNEL_SIZE = 7
BODY_ENHANCEMENT_THRESH_ITERATIONS = 1
BODY_ENHANCEMENT_KERNEL_OPEN_SIZE = 5
BODY_SKELETONIZE_IMAGE_KERNEL_SIZE2 = 5

BODY_SKELETONIZE_MULTIPLY_PHASE2 = 3

BODY_PHASE2_THRESHOLD = 120
BODY_PHASE2_THRESHOLD_PERCENT = 0.9
BODY_PHASE2_THRESHOLD_KERNEL_SIZE = 0

# === Width Sample Configuration ===
MAX_WIDTH_SAMPLE_COUNT = 100
MIN_WIDTH_SAMPLE_COUNT = 20
WIDTH_DISTANCE_SAMPLES_PER_PIXEL_AREA = 1 / 30
SAMPLE_SEED = 1


# === CLAHE Normalization Constants ===
NORMALIZE_VAR_RANGE = (650, 1200)
NORMALIZE_MEAN_RANGE = (60, 95)
NORMALIZE_ENTROPY_RANGE = (4.8, 6.0)

#small 654,61,4.85
#full (1391,104,6.8 ), (1400,93,6.7) , (1204,75,6.8)
#blob edge (1416,51,5.978)
#two colors (434,86,6.249)
#2.5 24*24 good 97% full (1391,104,6.8 ) 2.5 64*64 98%
#3 24*24 blob 90% full (1416,51,5.978)
# 1.3 6*6 good 97% small empty (654,61,4.85)

NORMALIZE_CLIP_RANGE = (1.3, 2.7)#(1.3, 2.0)              # Output range for clipLimit
NORMALIZE_TILE_SIZE_RANGE = (6, 128)#(32, 4)            # Output range for tileGridSize (from large to small tiles)

#tested:
#large full
# 1.5 64*64 90%
#1 8*8 too small for both cases
#1 32*32 too small for both
# 1.5 32*32 too small
# 3 32*32 good 95 for full, too many for empty
# 3 8*8 good 90 for full, bad for empty
# 3 16*16 good 80 for full
# 5 32*32 too many false positive for full
# 2.5 24*24 good 97 for full

#Small empty
#1.5 8*8 good 90 small
#2 8*8 bad small
#1.3 8*8 good 93
#1.1 8*8 too small
# 1.3 32*32 good 95
# 1.3 2*2 good  95
# 1.3 6*6 good 97


# ================= TissueAnalysis Class =================
class TissueAnalysis:
    def __init__(self):
        self._json_info = JsonInfo()

        # Processed image fields (set during processing)
        self._hem_pre_norm = None
        self._dab_pre_norm = None
        self._post_norm = None
        self._dab = None
        self._hem = None
        self._hem_binary1 = None
        self._hem_binary2 = None
        self._hem_phase2 = None
        self._body_mask = None
        self._cores_mask = None
        self._skeleton = None
        self._core_contours = None
        self._cores_idx = None
        self._arms_idx = None
        self._pre_arms_analysis = None
        self._sorted_cores = None
        self._analyzed_body_pre_mask = None
        self._analyzed_cores_pre_mask = None
        self._postFft = None
        self._results = None
        self._painted_bgr = None
        self._painted_bgr_skeleton = None

    # --- Read-only getters ---
    @property
    def dab(self):
        return self._dab

    @property
    def norm(self):
        return self._hem_pre_norm

    @property
    def hem(self):
        return self._hem

    @property
    def body_mask(self):
        return self._body_mask

    @property
    def cores_mask(self):
        return self._cores_mask

    @property
    def skeleton(self):
        return self._skeleton

    @property
    def core_contours(self):
        return self._core_contours

    @property
    def analyzed_body_pre_mask(self):
        return self._analyzed_body_pre_mask

    @property
    def analyzed_cores_pre_mask(self):
        return self._analyzed_cores_pre_mask

    @property
    def results(self):
        return self._results

    @property
    def painted_bgr(self):
        return self._painted_bgr

    @property
    def painted_bgr_skeleton(self):
        return self._painted_bgr_skeleton

    # --- JSON Logging Helper ---
    def _add_to_json(self, data: Any, category: str, sub_category: str = None,
                     id: str | int = None, should_print: bool = False):
        self._json_info.add_to_json(data, category, sub_category, id, should_print)

    # --- Small Helper Functions ---
    def _get_outer_contours(self, contours, hierarchy) -> Tuple[List[Any], Dict[int, Any]]:
        holes_by_outer = {}
        outer_contours = []
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for i, cnt in enumerate(contours):
                if hierarchy[i][3] == -1:
                    outer_contours.append((i, cnt))
                    holes_by_outer[i] = []
            for i, cnt in enumerate(contours):
                if hierarchy[i][3] != -1:
                    parent_idx = hierarchy[i][3]
                    if parent_idx in holes_by_outer:
                        holes_by_outer[parent_idx].append(cnt)
        return outer_contours, holes_by_outer

    def _sort_contours_by_area(self, contours, reverse=True):
        # Create top-left sorted ID mapping
        top_left_sorted = self._sort_contours_top_left(contours)
        contour_id_map = {id(c): idx for idx, c in enumerate(top_left_sorted)}

        # Sort by area
        contour_areas = [cv2.contourArea(c) for c in contours]
        pairs = list(zip(contour_areas, contours))
        sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=reverse)

        # Return tuples of (contour, top_left_id)
        return [(c, contour_id_map.get(id(c), -1)) for area, c in sorted_pairs]

    # def _sort_contours_by_area(self, contours, reverse=True):
    #     contour_areas = [cv2.contourArea(c) for c in contours]
    #     pairs = list(zip(contour_areas, contours))
    #     sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=reverse)
    #     return [c for area, c in sorted_pairs]

    def _sort_contours_top_left(self, contours):
        def contour_top_left(c):
            x, y, w, h = cv2.boundingRect(c)
            return (y, x)  # Top to bottom (y), then left to right (x)

        return sorted(contours, key=contour_top_left)

    # --- Channel Separation and Normalization ---
    def _separate_channels(self, bgr_image):
        # norm = normalize_image(bgr_image)
        # rgb = cv2.cvtColor(norm, cv2.COLOR_BGR2RGB)
        stain_sep = StainSeparation()
        hematoxylin, dab, _ = stain_sep.get_stain_channels(bgr_image)
        tile_grid_size, clip_limit, (variance, mean, entropy) = get_clahe_params(hematoxylin)
        #
        # dab_norm = normalize_image(dab, tile_grid_size, clip_limit )
        # hematoxylin_norm = normalize_image(hematoxylin, tile_grid_size, clip_limit )

        post_norm = normalize_image(bgr_image, (32,32), 8)
        hematoxylin, dab, _ = stain_sep.get_stain_channels(post_norm)

        self._post_norm = post_norm

        self._hem_pre_norm = hematoxylin
        self._dab_pre_norm = dab

        self._dab = dab#dab_norm
        self._hem = hematoxylin#hematoxylin_norm
        data = {
            "tile_grid_size" : tile_grid_size,
            "clip_limit" : clip_limit,
            "variance" : variance,
            "mean" : mean,
            "entropy" : entropy
        }
        self._add_to_json(data, "Normalize", "hematoxylin config",
                          should_print = True)


    def _separate_channels2(self, bgr_image):
        norm = normalize_image(bgr_image)
        rgb = cv2.cvtColor(norm, cv2.COLOR_BGR2RGB)
        stain_sep = StainSeparation()
        # get_stain_channels returns (hematoxylin, dab, residual)
        hematoxylin, dab, _ = stain_sep.get_stain_channels(rgb)
        self._hem_pre_norm = norm
        self._dab = dab
        self._hem = hematoxylin

    # --- Core Analysis ---
    def _analyze_cores(self):
        if USE_FFT:
            fft = fft_bandpass_filter(self._dab, FFT_LOW_FREQ, FFT_HIGH_FREQ)
        else:
            fft = self._dab
        gauss = cv2.GaussianBlur(fft, CORE_GAUSSIAN_KERNEL, CORE_GAUSSIAN_SIGMA)
        debug_info, binary = self._thresh_hist_close(
            gauss, CORE_PRIMARY_THRESHOLD, CORE_PRIMARY_THRESHOLD_PERCENT,
            CORE_THRESH_KERNEL_SIZE, CORE_THRESH_ITERATIONS, CORE_THRESH_KERNEL_OPEN_SIZE
        )
        self._hem_binary1 = binary

        self._add_to_json(debug_info, "_analyze_cores", "thresh1")
        debug_info, hem_binary = self._thresh_hist_close(
            self._hem, CORE_HEM_THRESHOLD, CORE_HEM_THRESHOLD_PERCENT, CORE_HEM_THRESHOLD_KERNEL_SIZE
        )
        self._hem_binary2 = hem_binary
        self._add_to_json(debug_info, "_analyze_cores", "hem_thresh")
        phase2 = np.where((binary != 0) & (hem_binary != 0), self._dab, 0)
        self._hem_phase2 = phase2

        debug_info, binary2 = self._thresh_hist_close(
            phase2, CORE_SECONDARY_THRESHOLD, CORE_SECONDARY_THRESHOLD_PERCENT, CORE_SECONDARY_THRESHOLD_KERNEL_SIZE
        )
        self._add_to_json(debug_info, "_analyze_cores", "thresh2")
        new_mask, sig_contours, _, _ = find_contours(binary2, MIN_CORE_AREA, MAX_CORE_AREA)
        if DEBUG_CORES:
            display_images(
                [self._dab, self._hem, binary, hem_binary, phase2, binary2, new_mask],
                "core phase"
            )
        self._analyzed_cores_pre_mask = binary2
        self._cores_mask = new_mask
        self._postFft = fft
        self._core_contours = sig_contours
        self._sorted_cores = self._sort_contours_by_area(self._core_contours)

    # --- Body Analysis ---
    def _analyze_dab_body(self):
        dab_mask = create_circles_mask(self._cores_mask, CORE_CIRCLE_RADIUS_MASK)
        masked_dab = np.where(dab_mask != 0, self._dab, 0)
        gaus = cv2.GaussianBlur(masked_dab, BODY_GAUSSIAN_KERNEL, BODY_GAUSSIAN_SIGMA)
        debug_info, binary1 = self._thresh_hist_close(
            gaus, BODY_SKELETON_THRESHOLD, BODY_SKELETON_THRESHOLD_PERCENT, BODY_SKELETON_THRESH_KERNEL_SIZE
        )
        self._add_to_json(debug_info, "_analyze_body", "thresh1")
        skeleton_local = skeletonize_image(binary1, BODY_SKELETONIZE_IMAGE_KERNEL_SIZE)
        arms_skeleton = skeleton_local & (~self._cores_mask)
        debug_info, binary2 = self._thresh_hist_close(
            gaus, BODY_ENHANCEMENT_THRESHOLD, BODY_ENHANCEMENT_THRESHOLD_PERCENT,
            BODY_ENHANCEMENT_THRESH_KERNEL_SIZE, BODY_ENHANCEMENT_THRESH_ITERATIONS, BODY_ENHANCEMENT_KERNEL_OPEN_SIZE
        )
        self._add_to_json(debug_info, "_analyze_body", "thresh1.5")
        skeleton2 = skeletonize_image(binary2, BODY_SKELETONIZE_IMAGE_KERNEL_SIZE2)
        gaus_aug = multiple_in_mask(masked_dab, skeleton2, BODY_SKELETONIZE_MULTIPLY_PHASE2)
        phase2_aug = cv2.bitwise_or(gaus_aug, arms_skeleton)
        debug_info, phase2_bin = self._thresh_hist_close(
            phase2_aug, BODY_PHASE2_THRESHOLD, BODY_PHASE2_THRESHOLD_PERCENT, BODY_PHASE2_THRESHOLD_KERNEL_SIZE
        )
        self._add_to_json(debug_info, "_analyze_body", "thresh2")
        if DEBUG_BODY_PHASE1:
            display_images([self._dab, gaus, binary1, binary2, gaus_aug, phase2_aug], "body phase1")
        if DEBUG_BODY_PHASE2:
            display_images([self._dab, masked_dab, gaus, binary1, binary2, gaus_aug, phase2_aug, phase2_bin], "body phase2")
        self._analyzed_body_pre_mask = phase2_bin

    # --- Important Cell Selection ---
    def _select_important_cells(self, body_mask: np.ndarray, important_mask: np.ndarray) -> np.ndarray:
        if body_mask.shape != important_mask.shape:
            raise ValueError("Masks must have the same dimensions.")
        # Connected components filtering.
        _, binary = None, body_mask
        num_labels, labels = cv2.connectedComponents(binary)
        imp_pixels = important_mask > 0
        imp_labels = labels[imp_pixels]
        important_bodies = np.unique(imp_labels)
        important_bodies = important_bodies[important_bodies != 0]
        new_mask = np.isin(labels, important_bodies).astype(np.uint8) * 255
        return new_mask

    # --- Combined Analysis ---
    def _analyse_cores_and_body(self, bgr_image):
        self._separate_channels(bgr_image)
        self._analyze_cores()
        self._analyze_dab_body()
        combined = cv2.bitwise_or(self._analyzed_body_pre_mask, self._cores_mask)
        rel_body = self._select_important_cells(combined, self._cores_mask)
        self._body_mask, _, _, _ = find_contours(rel_body, MIN_DAB_AREA, MAX_DAB_AREA)
        self._skeleton = skeletonize_image(self._body_mask)
        self._add_to_json(0, "_analyze_body", "filtered_small", should_print=False)
        self._add_to_json(0, "_analyze_body", "filtered_large", should_print=False)
        if DEBUG_BODY_AND_CORES:
            display_images(
                [self._dab, self._cores_mask, self._analyzed_body_pre_mask, combined, rel_body, self._body_mask, self._skeleton],
                "dab, cores, body, processed body, skeleton"
            )

    # --- Morphological Helpers ---
    def _close_open(self, img: np.ndarray, kernel_size: int, iter: int = 1,
                    kernel_open_size: int = -1) -> np.ndarray:
        if kernel_size > 0:
            kernel_open_size = kernel_size if kernel_open_size < 0 else kernel_open_size
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iter + 1)
        else:
            closed = img
        if kernel_open_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_open_size, kernel_open_size))
            return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iter)
        return closed

    def _thresh_close(self, img: np.ndarray, thresh: int, kernel_size: int,
                      iter: int = 1, kernel_open_size: int = -1) -> Tuple[int, np.ndarray]:
        opened = self._close_open(img, kernel_size, iter, kernel_open_size) if kernel_size > 0 else img
        _, binary = cv2.threshold(opened, thresh, 255, cv2.THRESH_BINARY)
        return thresh, binary

    def _thresh_hist_close(self, img, min_value: int, percent: float, kernel_size: int = 1,
                           iter: int = 1, kernel_open_size: int = -1) -> Tuple[List[Any], np.ndarray]:
        mask = (img != 0).astype(np.uint8)
        percent_value, hist, cumulative_hist = get_histogram_percentage(percent, img, mask=mask)
        threshold = max(min_value, percent_value)
        _, closed = self._thresh_close(img, threshold, kernel_size, iter, kernel_open_size)
        values = get_hist_values(img, mask)
        return [{"result": threshold, "percent": percent, "percent_value": percent_value, "min": min_value}, values], closed

    # --- Painted Image Creation ---
    def _create_painted_bgr(self, analysis_results: Dict[int, Dict[str, Any]], skeleton: np.ndarray,
                            bgr_image: np.ndarray, draw_skeleton: bool) -> np.ndarray:
        painted = bgr_image.copy()

        def rects_intersect(rect1, rect2) -> bool:
            x1, y1, w1, h1 = rect1
            x2, y2, w2, h2 = rect2
            br1 = (x1 + w1, y1 + h1)
            br2 = (x2 + w2, y2 + h2)
            return not (x1 > br2[0] or x2 > br1[0] or y1 > br2[1] or y2 > br1[1])

        relevant_core_holes = [d['related_holes'] for d in analysis_results.values()]
        body_contours = [d['body_contours'] for d in analysis_results.values()]

        if not draw_skeleton:
            cv2.drawContours(painted, body_contours, -1, (0, 0, 255), 2)
            for hole in relevant_core_holes:
                cv2.drawContours(painted, hole, -1, (50, 50, 50), cv2.FILLED)
        else:
            core_bbs = [d['core_bounding_box'] for d in analysis_results.values()]
            skel_cnts, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered = []
            for cnt in skel_cnts:
                cnt_bb = cv2.boundingRect(cnt)
                if any(rects_intersect(bb, cnt_bb) for bb in core_bbs):
                    filtered.append(cnt)
            cv2.drawContours(painted, filtered, -1, (255, 0, 255), 1)
        relevant_cores = [d['core_contour'] for d in analysis_results.values()]
        cv2.drawContours(painted, relevant_cores, -1, (0, 255, 0), 1)
        for cid, data in analysis_results.items():
            pos = tuple(data['core_center'] + np.array([30, 5]))
            cv2.putText(painted, f"{data['name']}: {int(data['core_area'])}",
                        pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
        return painted

    @staticmethod
    def create_colored_mask_with_ids(output_image, contour, label: str, color=(0, 255, 0)):
        cv2.drawContours(output_image, [contour], -1, color, thickness=cv2.FILLED)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])+ 10
            cy = int(M["m01"] / M["m00"])+ 10
            cv2.putText(output_image, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    # --- Arms Analysis ---
    def _analyze_arms(self, tile_row_col: Tuple[int, int], tile_x_y: Tuple[int, int],
                      min_core_body_diff: float, min_arms_length: float) -> Dict[int, Dict[str, Any]]:
        contours, hierarchy = cv2.findContours(self._body_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        outer_contours, holes_by_outer = self._get_outer_contours(contours, hierarchy)

        results = {}
        body_contour_missing_count = 0
        short_arms_count = 0
        small_arms_area_count = 0
        max_arm_length_for_valid_count = 0
        min_core_color_fail_count = 0


        self._cores_idx = np.zeros((*self._cores_mask.shape, 3), dtype=np.uint8)
        self._arms_idx = np.zeros((*self._cores_mask.shape, 3), dtype=np.uint8)

        self._pre_arms_analysis = np.zeros((*self._cores_mask.shape, 3), dtype=np.uint8)
        default_preview_color = (128, 128, 128)  # Gray

        for core_contour, idx in self._sorted_cores:
            self.create_colored_mask_with_ids(self._pre_arms_analysis, core_contour, str(idx), color=default_preview_color)

        type_to_color = {
            "zero-moments": (0, 0, 255),  # Red
            "min_core_color_fail": (0, 165, 255),  # Orange
            "body_missing": (255, 0, 0),  # Blue
            "small_arms_area": (255, 255, 0),  # Cyan
            "max_arm_length": (255, 0, 255),  # Magenta
            "valid": (0, 255, 0),  # Green
        }

        for sub_type, color in type_to_color.items():
            self._add_to_json(color, "analyze_arms_errors", sub_type + "-color")

        for core_contour, idx in self._sorted_cores:

            M = cv2.moments(core_contour)
            if M["m00"] == 0:
                self._add_to_json(idx, "analyze_arms_errors", "zero-moments")
                self.create_colored_mask_with_ids(self._cores_idx, core_contour, str(idx), color= type_to_color["zero-moments"])
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            core_center = (cx, cy)
            core_area = cv2.contourArea(core_contour)
            (cbb_x, cbb_y, cbb_w, cbb_h) = cv2.boundingRect(core_contour)
            core_bounding_box = (cbb_x, cbb_y, cbb_w, cbb_h)

            core_mask = np.zeros_like(self._cores_mask)
            cv2.drawContours(core_mask, [core_contour], -1, 255, thickness=cv2.FILLED)
            mean_val = cv2.mean(self._dab, mask=core_mask)[0]
            if mean_val < MIN_AVG_CORE_DAB_COLOR:
                min_core_color_fail_count += 1
                self._add_to_json({idx: mean_val}, "analyze_arms_errors", "min_core_color_fail")
                self.create_colored_mask_with_ids(self._cores_idx, core_contour, str(idx), color=type_to_color["min_core_color_fail"])
                continue

            relevent_body = None
            body_contour_id = None
            related_holes = []
            for outer_idx, body_contour in outer_contours:
                if cv2.pointPolygonTest(body_contour, core_center, False) >= 0:
                    relevent_body = body_contour
                    body_contour_id = outer_idx
                    related_holes = holes_by_outer.get(outer_idx, [])
                    break
            if relevent_body is None:
                body_contour_missing_count += 1
                self._add_to_json(idx, "analyze_arms_errors", "body_missing")
                self.create_colored_mask_with_ids(self._cores_idx, core_contour, str(idx), color=type_to_color[ "body_missing"])
                continue

            outer_contours.remove((body_contour_id, relevent_body))
            body_mask_local = np.zeros_like(self._body_mask)
            cv2.drawContours(body_mask_local, [relevent_body], -1, 255, thickness=cv2.FILLED)
            for hole in related_holes:
                cv2.drawContours(body_mask_local, [hole], -1, 0, thickness=cv2.FILLED)
            body_mask_local = body_mask_local.astype(np.uint8)
            core_mask = core_mask.astype(np.uint8)
            arms_mask = body_mask_local & (~core_mask)
            arms_area = np.count_nonzero(arms_mask)
            if arms_area < min_core_body_diff:
                small_arms_area_count += 1
                self._add_to_json(idx, "analyze_arms_errors", "small_arms_area")
                self.create_colored_mask_with_ids(self._cores_idx, core_contour, str(idx), color=type_to_color["small_arms_area"])
                continue

            skeleton_without_core = np.where(core_mask == 0, self._skeleton * body_mask_local, 0)
            total_arm_contours, _ = cv2.findContours(skeleton_without_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            arm_contours = [arm for arm in total_arm_contours if cv2.arcLength(arm, False) > min_arms_length]
            if len(arm_contours) < len(total_arm_contours):
                reduce_count = len(total_arm_contours) - len(arm_contours)
                short_arms_count += reduce_count
                self._add_to_json(reduce_count, "analyze_arms_errors", "reduce_arms", idx)
            arm_data = []
            for arm in arm_contours:
                arm_length = cv2.arcLength(arm, False)
                arm_data.append({'length': arm_length, 'contour': arm})

            for arm_idx, arm in enumerate(arm_contours):
                arm_label = f"{idx}_{arm_idx}"
                self.create_colored_mask_with_ids(self._arms_idx, arm, arm_label, color = type_to_color["valid"])


            arm_lengths = [d['length'] for d in arm_data]
            arm_contours_list = [d['contour'] for d in arm_data]
            num_arms = len(arm_data)
            max_arm_length = float(max(arm_lengths)) if arm_lengths else 0.0
            sum_arm_length = float(sum(arm_lengths)) if arm_lengths else 0.0
            avg_arm_length = float(np.average(arm_lengths)) if arm_lengths else 0.0

            if max_arm_length > MAX_ARM_LENGTH_FOR_VALID:
                max_arm_length_for_valid_count += 1
                self._add_to_json(idx, "analyze_arms_errors", "max_arm_length")
                self.create_colored_mask_with_ids(self._cores_idx, core_contour, str(idx), color=type_to_color["max_arm_length"])
                continue

            width_samples_count = int(WIDTH_DISTANCE_SAMPLES_PER_PIXEL_AREA * arms_area)
            width_samples_count = min(MAX_WIDTH_SAMPLE_COUNT, width_samples_count)
            width_samples_count = max(width_samples_count, MIN_WIDTH_SAMPLE_COUNT)
            self._add_to_json({"arms_area": arms_area, "width_samples_count": width_samples_count},
                              "analyze_arms", "width_samples_count")
            avg_width = (
                'Not_calculated'
                if not SAMPLE_WIDTH_FLAG
                else sample_widths_in_mask_distance(SAMPLE_SEED, arms_mask, width_samples_count)
                if arms_area != 0
                else 'NaN'
            )

            results[idx] = {
                'file': '',
                'name': f'{tile_row_col[0]}_{tile_row_col[1]}_{idx}',
                'arms/core': float(arms_area) / float(core_area) if core_area != 0 else 'NaN',
                'num_arms': num_arms,
                'average_arms_width': avg_width,
                'max_arm_length': max_arm_length,
                'avg_arm_length': avg_arm_length,
                'sum_arm_length': sum_arm_length,
                'arm_lengths': arm_lengths,
                'core_area': float(core_area),
                'body_area': float(core_area + arms_area),
                'arms_area': float(arms_area),
                'core/arms': float(core_area) / float(arms_area) if arms_area != 0 else 'NaN',
                'core_contour': core_contour,
                'arm_contours': arm_contours_list,
                'core_bounding_box': core_bounding_box,
                'core_center': core_center,
                'tile_row': tile_row_col[0],
                'tile_col': tile_row_col[1],
                'x': tile_x_y[0],
                'y': tile_x_y[1],
                'body_contours': relevent_body,
                'related_holes': related_holes,
                'core_w': cbb_w,
                'core_h': cbb_h,
                'total_bodies_count': 1,
            }
            self.create_colored_mask_with_ids(self._cores_idx, core_contour, str(idx), color=type_to_color["valid"])


        self._add_to_json(body_contour_missing_count, "analyze_arms_errors", "body_missing_count")
        self._add_to_json(small_arms_area_count, "analyze_arms_errors", "small_arms_area_count")
        self._add_to_json(short_arms_count, "analyze_arms_errors", "short_arms_count")
        self._add_to_json(max_arm_length_for_valid_count, "analyze_arms_errors", "long_arm_length_count")
        self._add_to_json(min_core_color_fail_count, "analyze_arms_errors", "min_core_color_fail_count")


        return results

    def _save_analyzed_arms_images(self, target_folder: str):
        debug_image(self._arms_idx, "arms_idx", target_folder)
        debug_image(self._cores_idx, "cores_idx", target_folder)
        debug_image(self._pre_arms_analysis, "pre_arms_analysis", target_folder)


    # --- Debug and Save Helpers ---
    def _save_analyzed_images(self, target_folder: str):
        debug_image(create_histogram_as_image(self._dab), "hist_dab", target_folder)
        debug_image(create_histogram_as_image(self._hem), "hist_hem", target_folder)
        debug_image(create_histogram_as_image(self._dab_pre_norm), "hist_dab_pre_norm", target_folder)
        debug_image(create_histogram_as_image(self._hem_pre_norm), "hist_hem_pre_norm", target_folder)
        debug_image(self._hem_pre_norm, "hem_pre_norm", target_folder)
        debug_image(self._dab_pre_norm, "dab_pre_norm", target_folder)
        debug_image(self._dab, "dab", target_folder)
        debug_image(self._hem, "hem", target_folder)
        debug_image(self._analyzed_body_pre_mask, "analyzed_body_pre_mask", target_folder)
        debug_image(self._analyzed_cores_pre_mask, "analyzed_core_pre_mask", target_folder)
        debug_image(self._body_mask, "body_mask", target_folder)
        debug_image(self._cores_mask, "cores_mask", target_folder)

        debug_image(self._hem_binary1, "analyzed_hem_binary1", target_folder)
        debug_image(self._hem_binary2, "analyzed_hem_binary2", target_folder)
        debug_image(self._hem_phase2, "analyzed_hem_phase2", target_folder)

    def _save_painted_images(self, target_folder: str):
        debug_image(self._painted_bgr, "painted_bgr", target_folder, True)
        debug_image(self._painted_bgr_skeleton, "painted_bgr_skeleton", target_folder)

    # --- Validation ---
    def _validate_image_std(self, tile_id, dab_img) -> bool:
        dabAvg, dabStd = histogram_avg_std(dab_img)
        self._add_to_json({"dabAvg": dabAvg, "dabStd": dabStd, "id": tile_id}, "dab_avg_std")
        return not (dabStd < MIN_DAB_STD_FOR_VALID_IMAGE)

    # --- Main Process ---
    def process_image(self, bgr_image, tile_row_col: Tuple[int, int], tile_x_y: Tuple[int, int],
                      target_folder: str, save_bgr_image: bool = False) -> None:
        start_time = time.perf_counter()
        if save_bgr_image:
            debug_image(bgr_image, "image", target_folder, True)
        self._analyse_cores_and_body(bgr_image)
        csv_path = os.path.join(target_folder, "results.csv")
        self._save_analyzed_images(target_folder)

        if self._core_contours is None or len(self._core_contours) == 0:
            self._add_to_json(0, "results_success_empty", "contour_count")
            save_results(None, csv_path)
            save_data_as_json(self._json_info.get_json(), os.path.join(target_folder, "info.json"))
            print(f"{tile_row_col} NoCoreCells: No results for {tile_row_col} empty file at path {target_folder}")
            return

        if not self._validate_image_std(tile_row_col, self._dab):
            self._add_to_json({}, "results_success_empty", "FlatImage", "std")
            save_results(None, csv_path)
            save_data_as_json(self._json_info.get_json(), os.path.join(target_folder, "info.json"))
            print(f"{tile_row_col} FlatImage: Image DAB doesn’t have color variance - no brown cells")
            return

        self._results = self._analyze_arms(tile_row_col, tile_x_y, MIN_CORE_BODY_DIFF, MIN_ARMS_LENGTH)
        self._save_analyzed_arms_images(target_folder)

        save_results(self._results, csv_path)

        self._painted_bgr = self._create_painted_bgr(self._results, self._skeleton, bgr_image, False)
        self._painted_bgr_skeleton = self._create_painted_bgr(self._results, self._skeleton, bgr_image, True)
        self._save_painted_images(target_folder)

        execution_time = format_time(time.perf_counter() - start_time)
        self._add_to_json(len(self._results), "results_success", "results_count")
        self._add_to_json(execution_time, "results_success", "time_elapsed")
        print(f"{tile_row_col}. Finished with {len(self._results)} after {execution_time} >> saved to {target_folder}")

        save_data_as_json(self._json_info.get_json(), os.path.join(target_folder, "info.json"))

def clamp(x, min_val, max_val):
    return max(min_val, min(x, max_val))
def lerp(a, b, t):
    return a + t * (b - a)
def get_clahe_params(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    variance = np.var(gray)
    mean = np.mean(gray)
    entropy = shannon_entropy(gray)

    var_norm = clamp((variance - NORMALIZE_VAR_RANGE[0]) / (NORMALIZE_VAR_RANGE[1] - NORMALIZE_VAR_RANGE[0]), 0.0,
                     1.0)
    mean_norm = clamp((mean - NORMALIZE_MEAN_RANGE[0]) / (NORMALIZE_MEAN_RANGE[1] - NORMALIZE_MEAN_RANGE[0]), 0.0,
                      1.0)
    ent_norm = clamp(
        (entropy - NORMALIZE_ENTROPY_RANGE[0]) / (NORMALIZE_ENTROPY_RANGE[1] - NORMALIZE_ENTROPY_RANGE[0]), 0.0,
        1.0)

    factor = (var_norm + mean_norm + ent_norm) / 3.0

    minClip = min(NORMALIZE_CLIP_RANGE)
    maxClip = max(NORMALIZE_CLIP_RANGE)

    clip_limit = clamp(lerp(NORMALIZE_CLIP_RANGE[0], NORMALIZE_CLIP_RANGE[1], factor), minClip,maxClip)

    tile_size = int(lerp(NORMALIZE_TILE_SIZE_RANGE[1], NORMALIZE_TILE_SIZE_RANGE[0], factor))

    minSize = min(NORMALIZE_TILE_SIZE_RANGE)
    maxSize = max(NORMALIZE_TILE_SIZE_RANGE)
    tile_size = clamp(int(tile_size / 2) * 2, minSize, maxSize)

    return (tile_size, tile_size), clip_limit, (variance, mean, entropy)