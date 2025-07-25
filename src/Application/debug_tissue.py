import datetime
import os
from typing import List

from src.Application.analyze_results import summarise_arms
from src.Application.tissue_analysis import TissueAnalysis
from src.Common.system_io import display_images
from src.Common.utils import load_image_debug

COMPARE_MULTI = False
INDEX = 4

BACKUP_FOLDER = "./backup_images/"
IMAGES_FOLDER = [
    "Tau_S10 31F11 #3 BR IHC for IBA-1_BF Color_1-41x21/3_4" #arms are identify too short //Image problem
    ,"Tau_S10 PBS #2 BR IHC for IBA-1_BF Color_1-48x37/3_5" # failed for small areas
    ,"Tau_S10 31F11 #2 BR IHC for IBA-1_BF Color_1-41x22/1_5"  # need smoothing
    ,"Tau_S10 PBS #2 BR IHC for IBA-1_BF Color_1-48x37/2_3" #edges artifacts
    ,"ARTE_S10 PBS #1 BR IBA-1_BF Color_1-43x24/2_4" ##lots with small linee
    ,"ARTE_S10 31F11 #3 BR IBA-1_BF Color_1-38x20/5_3" #full slice fail with percent
    ,"Tau_S10 31F11 #2 BR IHC for IBA-1_BF Color_1-41x22/2_4" # lots of examples
    ,"ARTE_S10 31F11 #3 BR IBA-1_BF Color_1-38x20/5_4" #small slice success
]
FOLDER = "/Users/doron/Workspace/Ido/data"
FILE_NAME = "image.tif"
IMAGES_PATH = [os.path.join(FOLDER, IMAGES_FOLDER[i], FILE_NAME) for i in range(len(IMAGES_FOLDER))]

SINGLE_IMAGE_PATH = IMAGES_PATH[INDEX]

def _multi_analysis_comparison():
    results = []

    #check all files exists
    for one_img in IMAGES_PATH:
        load_image_debug(one_img, BACKUP_FOLDER)

    for one_img in IMAGES_PATH:
        bgr_image = load_image_debug(one_img, BACKUP_FOLDER)
        results_folder = one_img.split(".")[0]
        ts = TissueAnalysis()
        report, [body_mask, cores_mask, painted_bgr,
                  painted_bgr_skeleton,
                  analyzed_body, analyzed_cores, dab, hem] = (
            ts.process_image(bgr_image, (0, 0), (0, 0), results_folder,
                             save_bgr_image=True))

        csv_summary = summarise_arms(report)
        print(f"results# {len(report)} analyzing the image done at {datetime.datetime.now()}")
        results.append({"img" : painted_bgr, "dab":dab, "csv_summary" : csv_summary})

    images:List[any] = [ result["img"] for result in results ]
    images2:List[any] = [ result["dab"] for result in results ]
    images.extend(images2)

    for csv_summary in [r["csv_summary"] for r in results]:
        print(csv_summary)

    display_images(images, "Multiple comparison")

def _single_analyze(debug = True):

    print(f"{datetime.datetime.now()} >> Reading the image  {SINGLE_IMAGE_PATH} ")

    results_folder =  SINGLE_IMAGE_PATH.split(".")[0]
    bgr_image = load_image_debug(SINGLE_IMAGE_PATH, BACKUP_FOLDER)

    ts =  TissueAnalysis()
    results, [body_mask, cores_mask, painted_bgr,
              _,
              analyzed_body, analyzed_cores, dab, hem] = (
        ts.process_image(bgr_image, (0,0), (0,0), results_folder ,
                      save_bgr_image = True))

    print (f"results# {len(results)} analyzing the image done at {datetime.datetime.now()}")

    if debug:
        summary = summarise_arms(results)
        print(f"__summary\n{summary}")
        ts._add_to_json(summary, "data_summary", should_print=False)
        display_images([painted_bgr, bgr_image, dab, hem,
                        analyzed_body, body_mask, analyzed_cores, cores_mask], f"'..{SINGLE_IMAGE_PATH[-45:]}' || final, org, body, core")

if __name__ == "__main__":
    if COMPARE_MULTI:
        _multi_analysis_comparison()
    else:
        _single_analyze()