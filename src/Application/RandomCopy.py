import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

def find_and_copy_images(
    root_dir,
    target_dir,
    target_filename,
    add_random_prefix=False,
    max_copies=None,
    random_prefix_digits=5
):
    root_path = Path(root_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    all_matches = defaultdict(list)

    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            if fname == target_filename:
                full_path = Path(dirpath) / fname
                rel_path = full_path.relative_to(root_path)
                first_depth = rel_path.parts[0]
                all_matches[first_depth].append(full_path)

    selected_files = []
    if max_copies:
        per_group = max(1, max_copies // max(1, len(all_matches)))
        for group in all_matches.values():
            selected_files.extend(group[:per_group])
        selected_files = selected_files[:max_copies]
    else:
        for group in all_matches.values():
            selected_files.extend(group)

    for src in selected_files:
        rel_parts = src.relative_to(root_path).parts
        clean_name = "_".join(rel_parts)
        if add_random_prefix:
            prefix = str(random.randint(10**(random_prefix_digits-1), 10**random_prefix_digits - 1))
            dest_name = f"{prefix}_{clean_name}"
        else:
            dest_name = clean_name

        dest_file = target_path / dest_name
        shutil.copy2(src, dest_file)

if __name__ == "__main__":
    find_and_copy_images("/Users/doron/Workspace/microglia_stain_ido/data",
                         "/Users/doron/Workspace/microglia_stain_ido/randome_data",
                         "painted_bgr.jpg", False, 200, 5)