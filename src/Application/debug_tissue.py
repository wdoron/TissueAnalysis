import datetime
import os
from typing import List

from src.Application.analyze_results import summarise_arms
from src.Application.tissue_analysis import TissueAnalysis
from src.Common.system_io import display_images
from src.Common.utils import load_image_debug

compare_multi = True
index = 4

backup_folder = "./backup_images/"
images_folder = [
    "Tau_S10 31F11 #3 BR IHC for IBA-1_BF Color_1-41x21/3_4" #arms are identify too short //Image problem
    ,"Tau_S10 PBS #2 BR IHC for IBA-1_BF Color_1-48x37/3_5" # failed for small areas
    ,"Tau_S10 31F11 #2 BR IHC for IBA-1_BF Color_1-41x22/1_5"  # need smoothing
    ,"Tau_S10 PBS #2 BR IHC for IBA-1_BF Color_1-48x37/2_3" #edges artifacts
    ,"ARTE_S10 PBS #1 BR IBA-1_BF Color_1-43x24/2_4" ##lots with small linee
    ,"ARTE_S10 31F11 #3 BR IBA-1_BF Color_1-38x20/5_3" #full slice fail with percent
    ,"Tau_S10 31F11 #2 BR IHC for IBA-1_BF Color_1-41x22/2_4" # lots of examples
    ,"ARTE_S10 31F11 #3 BR IBA-1_BF Color_1-38x20/5_4" #small slice success
]
folder = "/Users/doron/Workspace/Ido/data"
file_name = "image.tif"
images_path = [os.path.join(folder, images_folder[i], file_name)  for i in range(len(images_folder))]

single_image_path = images_path[index]

def _multi_analysis_comparison():
    results = []

    #check all files exists
    for one_img in images_path:
        load_image_debug(one_img, backup_folder)

    for one_img in images_path:
        bgr_image = load_image_debug(one_img, backup_folder)
        results_folder = one_img.split(".")[0]

        ts = TissueAnalysis()
        ts.process_image(bgr_image, (0, 0), (0, 0), results_folder,
                             save_bgr_image=True)

        csv_summary = summarise_arms(ts.results)
        print(f"results# {len(ts.results)} analyzing the image done at {datetime.datetime.now()}")
        results.append({"img" : ts.painted_bgr, "dab":ts.dab, "csv_summary" : csv_summary})

    images:List[any] = [ result["img"] for result in results ]
    images2:List[any] = [ result["dab"] for result in results ]
    images.extend(images2)

    for csv_summary in [r["csv_summary"] for r in results]:
        print(csv_summary)

    display_images(images, "Multiple comparison")

def _single_analyze(debug = True):

    print(f"{datetime.datetime.now()} >> Reading the image  {single_image_path} ")

    results_folder =  single_image_path.split(".")[0]
    bgr_image = load_image_debug(single_image_path, backup_folder)

    ts =  TissueAnalysis()
    ts.process_image(bgr_image, (0, 0), (0, 0), results_folder,
                     save_bgr_image=True)
    print (f"results# {len(ts.results)} analyzing the image done at {datetime.datetime.now()}")

    if debug:
        summary = summarise_arms(ts.results)
        print(f"__summary\n{summary}")
        ts._add_to_json(summary, "data_summary", should_print=False)
        display_images([ts.painted_bgr, bgr_image, ts.dab, ts.hem,
                        ts.analyzed_body_pre_mask, ts.body_mask, ts.analyzed_cores_pre_mask, ts.cores_mask], f"'..{single_image_path[-45:]}' || final, org, body, core")

if __name__ == "__main__":
    if compare_multi:
        _multi_analysis_comparison()
    else:
        _single_analyze()