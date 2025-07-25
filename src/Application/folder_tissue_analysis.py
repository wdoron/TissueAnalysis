import argparse
import datetime
import math
import os
import time
import traceback
from typing import Dict

import cv2
import matplotlib
from src.Application import tissue_analysis
from src.Application.analyze_results import remove_overlapping_cells, summarise_arms, save_results
from src.Application.tissue_analysis import TissueAnalysis
from src.Common.utils import format_time
matplotlib.use("Agg")  # Set the backend to not use the screen before importing pyplot
import concurrent.futures
from tifffile import imread

# default_path = "/Users/doron/Workspace/microglia_stain_ido/data"
default_path = "/Users/doron/Workspace/microglia_stain_ido/data/TestEdge"
default_file_name = None

tile_size = (4096, 4096)
max_pixels_length_of_body = 1024 #max length for overlapping tiles
max_tiles = None
debug_tile_row_col = None
use_multithreading = True

tissue_analysis.debug_cores = False
tissue_analysis.debug_body1 = False
tissue_analysis.debug_body2 = False
tissue_analysis.debug_body_and_cores = False
tissue_analysis.sample_width = True

skip_files = []

def call_process_image(tile, tile_row_col, tile_x_y, target_folder):
    if debug_tile_row_col is not None and debug_tile_row_col != tile_row_col:
        return tile_row_col, None
    ta = TissueAnalysis()
    ta.process_image(tile, tile_row_col, tile_x_y, target_folder, True)
    return tile_row_col, ta.results

def split_into_overlapping_tiles(image, tile_size=(2048, 2048), overlap=512):
    """
    Split the image into overlapping tiles.

    Parameters:
    - image: The input image as a NumPy array.
    - tile_size: Tuple indicating the (height, width) of each tile.
    - overlap: Number of pixels to overlap between tiles.

    Returns:
    - List of tile images.
    - List of corresponding top-left coordinates for each tile.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)#tif usually rgb
    img_height, img_width = image.shape[:2]
    tile_height, tile_width = tile_size

    step_y = tile_height - overlap
    step_x = tile_width - overlap

    tiles = []
    positions = []

    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            # Calculate the end coordinates
            end_y = y + tile_height
            end_x = x + tile_width

            # Adjust if the tile goes beyond the image boundary
            if end_y > img_height:
                end_y = img_height
                y = img_height - tile_height
            if end_x > img_width:
                end_x = img_width
                x = img_width - tile_width

            # Extract the tile
            tile = image[y:end_y, x:end_x].copy()
            tiles.append(tile)
            positions.append((y, x))

            # Break if we've reached the end
            if end_x == img_width:
                break
        if end_y == img_height:
            break

    return tiles, positions


def process_and_save(idx, tile, pos, path):
    """
    Process and save a single tile.

    Parameters:
    - idx: Index of the tile.
    - tile: The image tile.
    - pos: Position tuple (y, x).

    Returns:
    - Result string after processing.
    """
    tile_row_col = (
        idx // math.ceil(10000 / (2048 - 512)),  # Replace 10000 with actual image width if available
        idx % math.ceil(10000 / (2048 - 512))
    )
    tile_x_y = pos  # (start_y, start_x)
    row, col = tile_row_col
    x, y = tile_x_y
    target_folder = os.path.join(path, str(row) + "_" + str(col))
    if debug_tile_row_col is None or debug_tile_row_col == tile_row_col:
        os.makedirs(target_folder, exist_ok=True)
    id, results = call_process_image(tile, tile_row_col, tile_x_y, target_folder)
    return id, results


def analyze_file(path:str, file_name:str):

    analyse_path = os.path.join(path, file_name.split(".")[0])
    cdv_all_path = os.path.join(analyse_path, f"{file_name.split('.')[0]}_all.csv")
    image_path = os.path.join(path, file_name)

    image_folder_path = analyse_path #os.path.dirname(image_path)
    if not os.path.exists(image_folder_path):
        os.makedirs(image_folder_path)

    print(f"Reading the image... {datetime.datetime.now()}")
    try:
        image = imread(image_path)
    except Exception as e:
        print(f"Failed to read image from {image_path}: {e}")
        return

    if image is None:
        print(f"Failed to read image from {image_path}")
        return

    # Split the image into overlapping tiles
    print("Splitting the image into tiles...")
    tiles, positions = split_into_overlapping_tiles(image, tile_size, max_pixels_length_of_body)

    print(f"Total tiles created: {len(tiles)}")

    if max_tiles is not None and len(tiles) > max_tiles and debug_tile_row_col is None:
        tiles_len = len(tiles)
        tiles = tiles[:max_tiles]
        positions = positions[:max_tiles]
        print(f"Processing limited to first {max_tiles} / {tiles_len} tiles.")

    print("Processing tiles...")


    def process_and_save_wrapper(idx, tile, pos):
        try:
            return process_and_save(idx, tile, pos, image_folder_path)
        except Exception as exc:
            print(f"____Generated an exception!!!!! {exc}")
            traceback.print_exc()
            return None

    def process_all_tasks(tiles, positions, use_multithreading=True):
        if use_multithreading:
            num_cores = os.cpu_count()
            print(f"Number of CPU cores available: {num_cores}")

            with (concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor):
                task_iterator = executor.map(lambda args: process_and_save_wrapper(args[0], args[1][0], args[1][1]),
                             enumerate(zip(tiles, positions)))

                return _process_results(task_iterator)
        else:
            task_iterator = (
                process_and_save_wrapper(idx, tile, pos)
                for idx, (tile, pos) in enumerate(zip(tiles, positions))
            )
            return _process_results(task_iterator)

    def _process_results(task_iterator):
        results_list = []
        for result in task_iterator:
            if result is None:
                continue
            tile_row_col, dictionary_results = result
            if debug_tile_row_col is None or debug_tile_row_col == tile_row_col:
                results_count = len(dictionary_results) if dictionary_results is not None else 0
                print(
                    f"Processed {('found' if results_count != 0 else 'empty')}: {tile_row_col} results#: {results_count}")
                if results_count != 0:
                    results_list.extend(dictionary_results.values())
        return results_list

    summery = process_all_tasks(tiles, positions, use_multithreading=use_multithreading)

    print("Final Results List:")
    merged_results = remove_overlapping_cells(summery)

    ta = TissueAnalysis()
    summery = summarise_arms(merged_results)
    results_with_summery:Dict[int, Dict[str,any]] = {}
    results_with_summery[0] = summery
    for k,v in merged_results.items():
        results_with_summery[k+1] = v
    save_results(merged_results, cdv_all_path)

    print("All tiles processed or processing was terminated.")
    return summery, merged_results

def analyze_folder(folder_path, suffix = 'tif'):
    """
    Analyzes all files in the given folder that end with the given suffix.

    Checks in the following order:
      1. If folder_path is an absolute path and exists, use it.
      2. Else if folder_path exists as a relative path, use it.
      3. Else if a folder named 'Analyze' exists under the current working directory, use that.
      4. Else, call analyze_file with a default folder and file.
    """
    folder = None

    if  folder_path is not None:
        # 1. Check if the provided folder_path is an absolute path that exists.
        if os.path.isabs(folder_path) and os.path.exists(folder_path):
            folder = folder_path
        # 2. If not, check if folder_path exists relative to the current directory.
        elif os.path.exists(folder_path):
            folder = os.path.abspath(folder_path)

    if folder is None and os.path.exists(os.path.join(os.getcwd(), "Analyze")):
        folder = os.path.join(os.getcwd(), "Analyze")

    if folder is None:
        print("No valid folder found. Using default file.")
        folder = default_path
        if default_file_name is not None:
            file_name = default_file_name
            analyze_file(folder, file_name)
            return

    print(f"Analyzing folder: {folder}")
    sums:Dict[int, Dict[str,any]] = {}
    all_tables: Dict[int, Dict[str,any]] = {}
    for file in os.listdir(folder):
        if file.endswith(suffix):
            if file in skip_files:
                print (f"skip {file}")
                continue
            try:
                print(f"Analyzing file: {file}")
                start_time = time.perf_counter()
                sum, table = analyze_file(folder, file)
                sum['file'] = file
                for id, row in table.items():
                    row['file'] = file
                    all_tables[len(all_tables)] = row

                sums[len(sums)] = sum

                cdv_total_path = os.path.join(folder, f"sum.csv")
                save_results(sums, cdv_total_path)
                save_results(all_tables, folder + "/all_tables.csv")

                end_time = time.perf_counter()
                execution_time = format_time(end_time - start_time)
                print(f"finish with file {file} after {execution_time}")
            except BaseException as e:
                print (f"err with file {file} {e}")

def full_analysis():
    parser = argparse.ArgumentParser(
        description="Analyze files in a folder that match a specific suffix."
    )
    parser.add_argument("path", nargs='?', default=None, help="The folder path to analyze")
    args = parser.parse_args()

    analyze_folder(args.path)

if __name__ == "__main__":
    start_time = time.perf_counter()
    full_analysis()
    end_time = time.perf_counter()
    execution_time = format_time(end_time - start_time)
    print (f"\n_______Done all files after {execution_time}")

