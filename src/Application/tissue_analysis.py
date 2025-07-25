import os
import time
from collections.abc import Sequence
from scipy.ndimage import distance_transform_edt
import cv2
import numpy as np

from src.Application.analyze_results import save_results
from src.Common.histogram_utils import round_all, histogram_avg_std, create_histogram_as_image, \
    get_histogram_percentage
from src.Common.image_processing import skeletonize_image, find_contours, \
    normalize_image, fft_bandpass_filter, multiple_in_mask, get_hist_values
from src.Common.stain_separation import StainSeparation
from typing import Dict, Any, List, Tuple
from src.Common.system_io import display_images, save_data_as_json
from src.Common.utils import debug_image, format_time

#debugging
debug_cores = False
debug_body1 = False
debug_body2 = False
debug_body_and_cores = False

#Feature flags
sample_width = False

#cores analysis
min_deb_core_threshold_percent = 0.99
min_deb_core_threshold = 65
min_hem_core_threshold_percent = 0.9
min_hem_core_threshold = 160
hem_ellipsoid_median_approx_factor = 1.8
min_deb_core_threshold_percent2 = 0.3
min_deb_core_threshold2 = 101#150

#Body analysis
min_deb_threshold_percent_skeleton = 0.95
min_deb_threshold_skeleton = 70
min_deb_threshold_percent_enhacment = 0.86
min_deb_threshold_enhacment = 60
min_deb_threshold_percent_phase_2 = 0.9
min_deb_threshold2 = 120

#width sample configuration
max_width_sample_count = 100
min_width_sample_count = 20
width_distance_samples_per_pixel_area = 1/30

#ranges limits
dab_std_min_for_valid_image = 14

min_core_area = 75
max_core_area = 3000

min_dab_area = min_core_area
max_dab_area = 25000

cores_circle_radius_mask = 350

#remove conditions
min_avg_core_dab_color = 170
min_core_body_diff = 5
min_arms_length = 3
max_arm_length_for_valid = 556 #Above this value usually includes artifacts

class TissueAnalysis:
    def __init__(self):
        self._json = {}

    def _add_to_json(self, data: any, category: str | None, sub_category: str | None = None,
                     id: str | int | None = None, should_print: bool = False):
        # Apply default placeholders
        category = category or "NO_CAT"
        sub_category = sub_category if sub_category is not None else "NO_SUBCAT"
        id = id if id is not None else "NO_KEY"

        if category not in self._json:
            self._json[category] = {}

        if sub_category not in self._json[category]:
            self._json[category][sub_category] = {}

        if id not in self._json[category][sub_category]:
            self._json[category][sub_category][id] = []

        if isinstance(data, (list, tuple)):
            self._json[category][sub_category][id].extend(data)
        else:
            self._json[category][sub_category][id].append(data)

        if should_print:
            print(f"{category}.{sub_category}.{id} += {data}")
   
    def _sample_widths_in_mask_distance(self, seed_obj, image_mask: np.ndarray, samples: int) -> float:
        mask = image_mask.astype(bool)
        dt = distance_transform_edt(mask)
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return 0.0

        rng = np.random.RandomState(1)  #Create consistent random with the ID as seed
        indices = rng.choice(len(ys), size=samples, replace=True)
        sampled_dt = dt[ys[indices], xs[indices]]
        return 2 * np.mean(sampled_dt)


    def _analyze_arms(self,
                      dab_image: np.ndarray,
                      body: np.ndarray,
                      skeleton: np.ndarray,
                      cores: np.ndarray,
                      core_contours: Sequence[np.ndarray],
                      tile_row_col: Tuple[int, int],
                      tile_x_y: Tuple[int, int],
                      min_core_body_diff: float,
                      min_arms_length: float) -> Dict[int, Dict[str, Any]]:

       
        def get_outer_contours(contours, hierarchy ) -> Tuple[List[any], Dict[int,any]]:

            holes_by_outer = {}
            outer_contours = []
            if hierarchy is not None:
                hierarchy = hierarchy[0]
                for i, cnt in enumerate(contours):
                    if hierarchy[i][3] == -1:  # no parent means this is an outer contour
                        outer_contours.append((i, cnt))
                        holes_by_outer[i] = []
                for i, cnt in enumerate(contours):
                    if hierarchy[i][3] != -1:
                        parent_idx = hierarchy[i][3]
                        if parent_idx in holes_by_outer:
                            holes_by_outer[parent_idx].append(cnt)
            else:
                outer_contours = []
            return outer_contours, holes_by_outer

        contours, hierarchy = cv2.findContours(body, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        outer_contours, holes_by_outer = get_outer_contours(contours, hierarchy)

        results = {}
        body_contour_missing_count = 0
        short_arms_count = 0
        small_arms_area_count= 0
        max_arm_length_for_valid_count= 0
        min_core_color_fail_count = 0

        def _sort_contours_by_area(contours, reverse=True):
            contour_areas = [cv2.contourArea(contour) for contour in contours]
            contour_area_pairs = zip(contour_areas, contours)
            sorted_contours = sorted(contour_area_pairs, key=lambda x: x[0], reverse=reverse)
            return [contour for area, contour in sorted_contours]

        core_contours = _sort_contours_by_area(core_contours)
        for idx, core_contour in enumerate(core_contours):
            M = cv2.moments(core_contour)
            if M["m00"] == 0:
                self._add_to_json(idx, "analyze_arms_errors", "zero-moments", should_print = False)
                # print(f"Skipping core {idx} due to zero division in moments.")
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            core_center = (cx, cy)
            core_area = cv2.contourArea(core_contour)

            # (x, y, w, h) = cv2.boundingRect(core_contour)
            (cbb_x, cbb_y, cbb_w, cbb_h)  = cv2.boundingRect(core_contour)
            core_bounding_box = (cbb_x, cbb_y, cbb_w, cbb_h)

            core_mask = np.zeros_like(cores)
            cv2.drawContours(core_mask, [core_contour], -1, 255, thickness=cv2.FILLED)

            # Calculate the average intensity inside the contour area.
            mean_val = cv2.mean(dab_image, mask=core_mask)[0]

            # If the average intensity is below the threshold, remove this area from the mask.
            if mean_val < min_avg_core_dab_color:
                min_core_color_fail_count += 1
                self._add_to_json({idx : mean_val}, "analyze_arms_errors", "min_core_color_fail", should_print = False)
                continue

            relevent_body_contour = None
            body_contour_id = None

            related_holes = []
            for outer_idx, body_contour in outer_contours:
                # Use pointPolygonTest to see if the core center lies inside the body contour.
                if cv2.pointPolygonTest(body_contour, core_center, False) >= 0:
                    relevent_body_contour = body_contour
                    body_contour_id = outer_idx
                    related_holes = holes_by_outer.get(outer_idx, [])
                    break

            if relevent_body_contour is None:
                body_contour_missing_count += 1
                self._add_to_json(idx, "analyze_arms_errors", "body_missing", should_print = False)
                continue

            outer_contours.remove((body_contour_id, relevent_body_contour))# no duplicate cores in the same body here



            # --- Create the body mask properly, subtracting holes ---
            body_mask = np.zeros_like(body)
            cv2.drawContours(body_mask, [relevent_body_contour], -1, 255, thickness=cv2.FILLED)
            for hole in related_holes:
                cv2.drawContours(body_mask, [hole], -1, 0, thickness=cv2.FILLED)

            # arms_mask = cv2.bitwise_and(body_mask, cv2.bitwise_not(core_mask))
            # arms_area = cv2.countNonZero(arms_mask)

            body_mask = body_mask.astype(np.uint8)  # Ensure correct dtype
            core_mask = core_mask.astype(np.uint8)

            arms_mask = body_mask & (~core_mask)  # Optimized bitwise operation
            arms_area = np.count_nonzero(arms_mask)  # Faster count

            if arms_area < min_core_body_diff:
                small_arms_area_count += 1
                self._add_to_json(idx, "analyze_arms_errors", "small_arms_area", should_print = False)
                continue

                # max_arm_length = 7000
                # min_bounding_box_ratio_to_core = .5

            # Remove core from the skeleton.
            # @OPTIMIZATION
            # masked_skeleton = cv2.bitwise_and(skeleton, body_mask)
            # skeleton_without_core = cv2.bitwise_and(masked_skeleton, cv2.bitwise_not(core_mask))
            skeleton_without_core = np.where(core_mask == 0, skeleton * body_mask, 0)

            total_arm_contours, _ = cv2.findContours(skeleton_without_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter out arms that are too short.
            arm_contours = [arm for arm in total_arm_contours if cv2.arcLength(arm, False) > min_arms_length]

            if len(arm_contours) < len(total_arm_contours):
                reduce_arms_count = len(total_arm_contours) - len(arm_contours)
                short_arms_count += reduce_arms_count
                self._add_to_json(reduce_arms_count, "analyze_arms_errors", "reduce_arms", idx, should_print = False)

                # Optionally print or log the number of too short arms.
                # print(f"Removed {len(total_arm_contours) - len(arm_contours)} too small arm contours.")

            arm_data = []
            for arm_contour in arm_contours:
                arm_length = cv2.arcLength(arm_contour, False)

                arm_data.append({
                    'length': arm_length,
                    'contour': arm_contour
                })


            arm_lengths = [arm['length'] for arm in arm_data]
            # arm_endpoints = [arm['endpoint'] for arm in arm_data]
            arm_contours_list = [arm['contour'] for arm in arm_data]
            num_arms = len(arm_data)
            max_arm_length = float(max(arm_lengths)) if arm_lengths else 0.0
            sum_arm_length = float(sum(arm_lengths)) if arm_lengths else 0.0
            avg_arm_length = float(np.average(arm_lengths)) if arm_lengths else 0.0

            if max_arm_length > max_arm_length_for_valid:
                max_arm_length_for_valid_count += 1
                self._add_to_json(idx, "analyze_arms_errors", "max_arm_length", should_print = False)
                continue

            width_samples_count = int(width_distance_samples_per_pixel_area * arms_area)
            width_samples_count = min(max_width_sample_count, width_samples_count)
            width_samples_count = max(width_samples_count, min_width_sample_count)
            self._add_to_json({"arms_area" : arms_area, "width_samples_count" : width_samples_count}, "analyze_arms", "width_samples_count", should_print=False)

            avg_width = 'Not_calculated' if not sample_width else self._sample_widths_in_mask_distance(tile_row_col, arms_mask, width_samples_count) if arms_area != 0 else 'NaN'

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
                'body_contours': relevent_body_contour,
                'related_holes': related_holes,
                'core_w': cbb_w,
                'core_h': cbb_h,
                'total_bodies_count': 1,
            }

        if (body_contour_missing_count > 0
                or short_arms_count > 0
                or small_arms_area_count > 0
                or max_arm_length_for_valid_count > 0
                or min_core_color_fail_count):

            self._add_to_json(body_contour_missing_count, "analyze_arms_errors", "body_missing_count", should_print = False)
            self._add_to_json(small_arms_area_count, "analyze_arms_errors", "small_arms_area_count", should_print = False)
            self._add_to_json(short_arms_count, "analyze_arms_errors", "short_arms_count", should_print = False)
            self._add_to_json(max_arm_length_for_valid_count, "analyze_arms_errors", "long_arm_length_count", should_print = False)
            self._add_to_json(min_core_color_fail_count, "analyze_arms_errors", "min_core_color_fail_count", should_print = False)

            # print(f"Some data was removed: body_missing={body_contour_missing_count} "
            #       f"short_arms={short_arms_count} small_arm_area={small_arms_area_count} "
            #       f"max_arm_length_for_valid_count={max_arm_length_for_valid_count} " +
            #       f"min_core_color_fail_count={min_core_color_fail_count} ")

        return results

    def _select_important_cells(self, body_n_nuclias_mask, important_nuclias_mask):
        """
        Creates a new mask that includes only the bodies from body_n_nuclias_mask
        that contain at least one important nucleus from important_nuclias_mask.

        Parameters:
        - body_n_nuclias_mask (numpy.ndarray): Grayscale binary mask of bodies with nuclei.
        - important_nuclias_mask (numpy.ndarray): Grayscale binary mask of important nuclei.

        Returns:
        - numpy.ndarray: Grayscale binary mask with only important bodies.
        """
        # Validate Input Shapes
        if body_n_nuclias_mask.shape != important_nuclias_mask.shape:
            raise ValueError("Both masks must have the same dimensions.")

        _, body_binary = None, body_n_nuclias_mask# cv2.threshold(body_n_nuclias_mask, 127, 255, cv2.THRESH_BINARY)
        _, important_binary = None, important_nuclias_mask#cv2.threshold(important_nuclias_mask, 127, 255, cv2.THRESH_BINARY)

        num_labels, labels = cv2.connectedComponents(body_binary)

        important_pixels = important_binary > 0  # Boolean mask

        important_labels = labels[important_pixels]

        important_bodies = np.unique(important_labels)
        important_bodies = important_bodies[important_bodies != 0]

        new_mask = np.isin(labels, important_bodies).astype(np.uint8) * 255

        return new_mask

    
    def _channels_transform(self, bgr_image):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        stain_separator = StainSeparation()
        hematoxylin, dab, residual_channel = stain_separator.get_stain_channels(rgb_image)
        return dab, hematoxylin

   


   
    def _close_open(self, img: np.ndarray, kernel_size: int, iter: int = 1, kernel_open_size:int = -1) -> Tuple[int, np.ndarray]:
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

   
    def _thresh_close(self, img: np.ndarray, thresh: int, kernel_size: int, iter: int = 1, kernel_open_size:int = -1) -> Tuple[int, np.ndarray]:
        if kernel_size > 0:
            opened = self._close_open(img, kernel_size, iter, kernel_open_size)
        else:
            opened = img
        _, binary = cv2.threshold(opened, thresh, 255, cv2.THRESH_BINARY)
        return thresh, binary

   
    def _thresh_hist_close(self, img, min_value, percent, kernel_size: int = 1, iter: int = 1, kernel_open_size:int = -1)\
            -> Tuple[Tuple[float, float, float, Dict[float, float]], np.ndarray]:
        mask = (img != 0).astype(np.uint8)

        percent_value, hist, cumulative_hist = get_histogram_percentage(percent, img, mask = mask)
        threshold = max(min_value, percent_value)
        _, closed = self._thresh_close(img, threshold, kernel_size, iter, kernel_open_size)
        values = get_hist_values(img, mask)
        return [{"result":threshold, "percent":percent , "percent_value" : percent_value, "min" : min_value}, values], closed

    def _analyze_cores(self, dab_image, hem_image):
        fft = fft_bandpass_filter(dab_image, 10, 300)
        gauss = cv2.GaussianBlur(fft, (21,21), 43)

        debug_info, binary = self._thresh_hist_close(gauss, min_deb_core_threshold, min_deb_core_threshold_percent, 9, 1,0 )
        self._add_to_json(debug_info, "_analyze_cores", "thresh1")

        if min_hem_core_threshold_percent != 0:
            debug_info, hem_binary = self._thresh_hist_close(hem_image, min_hem_core_threshold, min_hem_core_threshold_percent, 3)
            self._add_to_json(debug_info, "_analyze_cores", "hem_thresh")
            phase2 = np.where((binary != 0) & (hem_binary != 0), dab_image, 0)
        else:
           hem_binary = np.zeros_like(binary)
           phase2 = np.where(binary != 0, dab_image, 0)

        debug_info, binary2 = self._thresh_hist_close(phase2, min_deb_core_threshold2, min_deb_core_threshold_percent2, 0)
        self._add_to_json(debug_info, "_analyze_cores", "thresh2")
        new_mask, significant_contours, _, skipped = find_contours(binary2, min_core_area, max_core_area)
        # if (len(skipped) > 0):
        #     print(f"core contours skipped {len(skipped)} left {len(significant_contours)}")

        if debug_cores:
            display_images([dab_image, hem_image, binary, hem_binary, phase2, binary2, new_mask], "core phase")

        return binary2, new_mask, significant_contours

    @staticmethod
    def _create_circles_mask_from_cores(core_mask: np.ndarray, radius: int = 500) -> np.ndarray:
        _, binary = cv2.threshold(core_mask, 127, 255, cv2.THRESH_BINARY)
        circles_mask = np.zeros_like(binary)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(circles_mask, (cx, cy), radius, 255, thickness=-1)
            else:
                continue

        return circles_mask

   
    def _analyze_dab_body(self, dab_image, core_mask):

        dab_mask = self._create_circles_mask_from_cores(core_mask, cores_circle_radius_mask)
        masked_dab = np.where(dab_mask != 0, dab_image, 0)
        gaus = cv2.GaussianBlur(masked_dab, (31,31), 50)

        #skeleton 1
        debug_info, binary1 = self._thresh_hist_close(gaus, min_deb_threshold_skeleton, min_deb_threshold_percent_skeleton, 5)
        self._add_to_json(debug_info, "_analyze_body", "thresh1")
        skeleton = skeletonize_image(binary1, 3)
        arms_skeleton_mask = skeleton & (~core_mask)  # Optimized bitwise operation

        # skeleton 2
        debug_info2, binary2 = self._thresh_hist_close(gaus,
                                                       min_deb_threshold_enhacment,
                                                       min_deb_threshold_percent_enhacment,
                                                       7, 1, 5)
        self._add_to_json(debug_info2, "_analyze_body", "thresh1.5")
        skeleton2 = skeletonize_image(binary2, 4)#Create thick areas to fix connection

        #merge
        gause_augmented  = multiple_in_mask(masked_dab, skeleton2, 3)
        phase2_augmented = cv2.bitwise_or(gause_augmented, arms_skeleton_mask)

        debug_info2, phase2_binary = self._thresh_hist_close(phase2_augmented, min_deb_threshold2, min_deb_threshold_percent_phase_2, 0)
        self._add_to_json(debug_info2, "_analyze_body", "thresh2")

        if debug_body1:
            display_images([dab_image, gaus, binary1, binary2, gause_augmented, phase2_augmented], "body phase1")
        if debug_body2:
            display_images([dab_image, masked_dab, gaus, binary1, binary2,  gause_augmented, phase2_augmented, phase2_binary], "body phase2")
        return phase2_binary

    def _analyse_dab_hema(self, bgr_image):
        dab, hem = self._channels_transform(bgr_image)

        analyzed_cores_pre_mask, cores_mask, core_contours = self._analyze_cores(dab, hem)#analyze_hem_cores(hem)
        analyzed_body = self._analyze_dab_body(dab, cores_mask)

        combined_body = cv2.bitwise_or(analyzed_body, cores_mask)

        relevant_body_image = self._select_important_cells(combined_body, cores_mask)
        relevant_body_mask, body_contours, body_contour_heirarchy, [small,large] = find_contours(relevant_body_image, min_dab_area, max_dab_area)

        self._add_to_json(len(small), "_analyze_body", "filtered_small", should_print = False)
        self._add_to_json(len(large), "_analyze_body", "filtered_large", should_print = False)

        skeleton = skeletonize_image(relevant_body_mask)

        if debug_body_and_cores:
            display_images([ dab, cores_mask, analyzed_body, combined_body, relevant_body_image, relevant_body_mask, skeleton], "dab, cores, body, ...processed body..., skeleton")

        return (relevant_body_mask, cores_mask, skeleton,
                [body_contours, body_contour_heirarchy],
                core_contours, analyzed_body, analyzed_cores_pre_mask, dab, hem)

    def _validate_image_std(self, id, dab):
        (dabAvg, dabStd) = round_all(histogram_avg_std(dab), 1)
        self._add_to_json({"dabAvg": dabAvg, "dabStd": dabStd, "id": id}, "dab_avg_std", should_print=False)
        return not (dabStd < dab_std_min_for_valid_image)

    def process_image(self, bgr_image, tile_row_col, tile_x_y, target_folder, save_bgr_image=False) -> Tuple[Dict[int,Dict[str,any]],List[np.ndarray]]:

        start_time = time.perf_counter()

        if save_bgr_image:
            debug_image(bgr_image, "image", target_folder, True)

        norm_image = normalize_image(bgr_image)

        body_mask, cores_mask, skeleton, dab_contours_n_hier, core_contours, analyzed_body_pre_mask, analyzed_core_pre_mask, dab, hem = (
            self._analyse_dab_hema(norm_image))
        csv_path =  os.path.join(target_folder, "results.csv")

        debug_image(create_histogram_as_image(dab), "hist_dab", target_folder)
        debug_image(create_histogram_as_image(hem), "hist_hem", target_folder)
        debug_image(dab, "dab", target_folder)
        debug_image(hem, "hem", target_folder)
        debug_image(analyzed_body_pre_mask, "analyzed_body_pre_mask", target_folder)
        debug_image(analyzed_core_pre_mask, "analyzed_core_pre_mask", target_folder)
        debug_image(body_mask, "body_mask", target_folder)
        debug_image(cores_mask, "cores_mask", target_folder)

        if len(dab_contours_n_hier[0]) == 0:
            self._add_to_json(0, "results_success_empty", "contour_count", should_print = False)

            save_results(None, csv_path)
            save_data_as_json(self._json, os.path.join(target_folder, "info.json"))

            txt = f"{tile_row_col} NoCoreCells: No results for {tile_row_col} empty file at path {target_folder}"
            print(txt)

            return None, [body_mask, cores_mask, None, None, analyzed_body_pre_mask, analyzed_core_pre_mask, dab, hem]

        result = {}
        if not self._validate_image_std(tile_row_col, dab):
            self._add_to_json(result, "results_success_empty", "FlatImage", "std", should_print = False)

            save_results(None, csv_path)
            save_data_as_json(self._json, os.path.join(target_folder, "info.json"))

            print(f"{tile_row_col} FlatImage: Image DAB doesnt have color variance - no brown cells")
            return None, [body_mask, cores_mask, None, None, analyzed_body_pre_mask, analyzed_core_pre_mask, dab, hem]

       
        def create_painted_bgr(analysis_results, skeleton, bgr_image, draw_skeleton):
            bgr_image_with_skeleton = bgr_image.copy()
            def rects_intersect(rect1, rect2):
                x1, y1, w1, h1 = rect1
                x2, y2, w2, h2 = rect2
                br1 = (x1 + w1, y1 + h1)
                br2 = (x2 + w2, y2 + h2)
                if x1 > br2[0] or x2 > br1[0]:
                    return False
                if y1 > br2[1] or y2 > br1[1]:
                    return False
                return True

            relevant_core_holes = [data['related_holes'] for data in analysis_results.values()]
            body_contours = [data['body_contours'] for data in analysis_results.values()]

            if not draw_skeleton:
                cv2.drawContours( bgr_image_with_skeleton, body_contours, -1, (0, 0, 255),2)

                for hole in relevant_core_holes:
                    cv2.drawContours(bgr_image_with_skeleton, hole, -1, (50,50,50), cv2.FILLED)

            if draw_skeleton:
                core_bbs = [data['core_bounding_box'] for data in analysis_results.values()]

                skeleton_contours, hierarchy = cv2.findContours(
                    skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                filtered_skeleton_contours = []
                for cnt in skeleton_contours:
                    cnt_bb = cv2.boundingRect(cnt)
                    if any(rects_intersect(core_bb, cnt_bb) for core_bb in core_bbs):
                        filtered_skeleton_contours.append(cnt)
                cv2.drawContours( bgr_image_with_skeleton, filtered_skeleton_contours, -1, (255, 0, 255), 1)

            relevant_cores_contours = [data['core_contour'] for data in analysis_results.values()]
            cv2.drawContours( bgr_image_with_skeleton, relevant_cores_contours, -1, (0, 255, 0),1)


            for core_id, data in analysis_results.items():
                label_pos = tuple(data['core_center'] + np.array([30, 5]))
                cv2.putText(bgr_image_with_skeleton, data['name'] + ": " + str(int(data['core_area'])),
                            label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (255, 0, 0), 1)

            # Optional: Debug output
            return bgr_image_with_skeleton

        results = self._analyze_arms(dab, body_mask, skeleton, cores_mask, core_contours, tile_row_col, tile_x_y,
                                     min_core_body_diff, min_arms_length) # dab_contours_n_hier[0]
        save_results(results, csv_path)
        painted_bgr = create_painted_bgr(results, skeleton, bgr_image, False)
        debug_image(painted_bgr, "painted_bgr", target_folder, True)
        painted_bgr_skeleton = create_painted_bgr(results, skeleton, bgr_image, True)
        debug_image(painted_bgr_skeleton, "painted_bgr_skeleton", target_folder)

        end_time = time.perf_counter()
        execution_time = format_time(end_time - start_time)

        print(f"{tile_row_col}. Finished, found  {len(results)} after {execution_time} >>  saved to {target_folder}")
        self._add_to_json(len(results), "results_success", "results_count", should_print = False)
        self._add_to_json(execution_time, "results_success", "time_elapsed", should_print = False)
        save_data_as_json(self._json, os.path.join(target_folder, "info.json"))
        return results, [body_mask, cores_mask, painted_bgr, painted_bgr_skeleton, analyzed_body_pre_mask, analyzed_core_pre_mask, dab, hem]