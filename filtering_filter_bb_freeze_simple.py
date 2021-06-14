import json
import os
from glob import glob
import numpy as np
import argparse
import sys
from warnings import warn
import shutil
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument("base_folder", type=str, default=".", help="The folder containing either the data directly or folders with data.")
parser.add_argument("--same-frame-limit", "-s", default=2, type=int, help="The number of consecutive frames with the same pose after which the next frame(s) becomes invalid.")
parser.add_argument("--list-valid", "-l", action="store_true", default=False, help="Output a list of valid frames.")
file_handling_group = parser.add_mutually_exclusive_group()
file_handling_group.add_argument("--exile-invalid", "-e", action="store_true", default=False, help="Move invalid BBoxes to a separate folder.")
file_handling_group.add_argument("--copy-valid", "-c", action="store_true", default=False, help="Copy valid BBoxes to a separate folder.")
args = parser.parse_args()

base_folder = args.base_folder
if not (os.path.exists(base_folder) and os.path.isdir(base_folder)):
    raise IOError("Ivalid base folder!")

same_frame_limit = args.same_frame_limit
bf_content = os.listdir(base_folder)
if "6dof" in (f.lower() for f in bf_content):  # processing of single folder
    if base_folder[-1] == os.path.sep:
        base_folder = base_folder[:-1]
    base_folder, folder_name = os.path.split(base_folder)
    set_folders = [folder_name]
else:  # processing of multiple folders
    set_folders = bf_content


def convertStamp(stamp, base_sec):
    nsec = "nsec" if "nsec" in stamp.keys() else "nsecs"
    return (stamp["sec"] - base_sec) + stamp[nsec] * 1e-9


def comparePose(a, b):
    return all([a["translation"][field] == b["translation"][field] for field in "xyz"] + [a["rotation"][field] == b["rotation"][field] for field in "xyzw"])


for folder_name in set_folders:
    folder = os.path.join(base_folder, folder_name)
    folder_6dof = os.path.join(folder, "6DOF")
    folder_bb = os.path.join(folder, "BBox")
    jsons = sorted(glob(os.path.join(folder_6dof, "*.json")))

    last_pose_C1 = None
    last_pose_C2 = None
    invalid_C1_count = 0
    invalid_C2_count = 0
    base_sec = 0
    valid = []
    invalid = []
    infoDict = {}
    for jj in jsons:
        with open(jj, "r") as f:
            d = json.load(f)
        frame_name = next(iter(d.keys()))
        bb_file = os.path.join(folder_bb, frame_name + ".json")
        if not os.path.isfile(bb_file):
            continue
        with open(bb_file, "r") as f:
            bd = json.load(f)
        frame_name = next(iter(bd.keys()))

        is_valid = True
        frame = bd[frame_name]
        poseC1 = frame["tracker_to_C1"]
        poseC2 = frame["tracker_to_C2"]
        if not base_sec:
            base_sec = poseC1["stamp"]["sec"]

        stamp_c1, stamp_c2 = convertStamp(poseC1["stamp"], base_sec), convertStamp(poseC2["stamp"], base_sec)
        infoDict[frame_name] = {
            "time_c1": stamp_c1,
            "stamp_c1": poseC1["stamp"],
            "time_c2": stamp_c2,
            "stamp_c2": poseC2["stamp"],
            "valid": True,
            "reason": ""
        }

        if last_pose_C1 is None:
            last_pose_C1 = poseC1
        else:
            if comparePose(last_pose_C1, poseC1):
                invalid_C1_count += 1
                if invalid_C1_count > same_frame_limit:
                    infoDict[frame_name]["reason"] = "pose lapse"
                    is_valid = False
            else:
                invalid_C1_count = 0
                last_pose_C1 = poseC1
        if last_pose_C2 is None:
            last_pose_C2 = poseC2
        else:
            if comparePose(last_pose_C2, poseC2):
                invalid_C2_count += 1
                if invalid_C2_count > same_frame_limit:
                    infoDict[frame_name]["reason"] = "pose lapse"
                    is_valid = False
            else:
                invalid_C2_count = 0
                last_pose_C2 = poseC2

        if is_valid:
            valid.append(frame_name)
        else:
            infoDict[frame_name]["valid"] = False
            if not infoDict[frame_name]["reason"]:
                infoDict[frame_name]["reason"] = "time diff"
            invalid.append(bb_file)

    dt = datetime.now()
    file_stamp = dt.strftime("%Y-%m-%d_%H_%M_%S")
    nice_stamp = dt.strftime("%Y-%m-%d @ %H:%M:%S")
    with open(os.path.join(folder, f"delay_filter_report_{file_stamp}.txt"), "w+") as f:
        f.writelines("* BBox vs Image time filtering report *\n")
        f.writelines(f"Folder name: {folder_name}\n")
        f.writelines(f"Time: {nice_stamp}\n")
        f.writelines("Arguments:\n")
        for k, v in args.__dict__.items():
            if "base_folder" in k:
                continue
            f.writelines(f"\t{k}: {v}\n")
        f.writelines("Parameters:\n")
        f.writelines("Stats:\n")
        f.writelines(f"\tvalid: {len(valid)}\n")
        f.writelines(f"\tinvalid: {len(invalid)}\n")
        f.writelines(f"\ttotal: {len(valid) + len(invalid)}\n")

        f.writelines("\nInvalid frames:\n")
        for frame, infd in infoDict.items():
            if infd["valid"]:
                continue
            f.writelines(f"- {frame}\n")
            f.writelines(f"\treason: {infd['reason']}\n")

            f.writelines(f"\tc1 stamp: {infd['stamp_c1']['sec']}.{infd['stamp_c1']['nsec']:09d}\n")
            diff = infd['time_c1']
            f.writelines(f"\tc1 time: {infd['time_c1']:.04f} ({'+' if diff >= 0 else ''}{diff:.04f})\n")

            f.writelines(f"\tc2 stamp: {infd['stamp_c2']['sec']}.{infd['stamp_c2']['nsec']:09d}\n")
            diff = infd['time_c2']
            f.writelines(f"\tc2 time: {infd['time_c2']:.04f} ({'+' if diff >= 0 else ''}{diff:.04f})\n")


    if args.exile_invalid:  # move invalid
        folder_bb_vis = os.path.join(folder, "BBox_visualization")
        if os.path.isdir(folder_bb_vis):
            folder_bb_vis_invalid = os.path.join(folder, "BBox_visualization_invalid")
            os.mkdir(folder_bb_vis_invalid)
            for f in invalid:
                bp, ff = os.path.split(f)
                ff, _ = os.path.splitext(ff)
                fc1, fc2 = os.path.join(folder_bb_vis, f"C1{ff}.png"), os.path.join(folder_bb_vis, f"C2{ff}.png")
                
                try:
                    shutil.move(fc1, folder_bb_vis_invalid)                    
                except:                    
                    continue
                
                try:                
                    shutil.move(fc2, folder_bb_vis_invalid)                    
                except:                    
                    continue
                
        folder_bb_invalid = os.path.join(folder, "BBox_invalid")
        os.mkdir(folder_bb_invalid)
        for f in invalid:
            shutil.move(f, folder_bb_invalid)
    elif args.copy_valid:  # copy valid
        folder_bb_vis = os.path.join(folder, "BBox_visualization")
        if os.path.isdir(folder_bb_vis):
            folder_bb_vis_valid = os.path.join(folder, "BBox_visualization_valid")
            os.mkdir(folder_bb_vis_valid)
            for f in valid:
                bp, ff = os.path.split(f)
                ff, _ = os.path.splitext(ff)
                fc1, fc2 = os.path.join(folder_bb_vis, f"C1{ff}.png"), os.path.join(folder_bb_vis, f"C2{ff}.png")
                
                try:
                    shutil.copy(fc1, folder_bb_vis_valid)
                except:
                    continue
                
                try:                
                    shutil.copy(fc2, folder_bb_vis_valid)
                except:
                    continue
                
        folder_bb_valid = os.path.join(folder, "BBox_valid")
        os.mkdir(folder_bb_valid)
        for f in valid:
            shutil.copy(os.path.join(folder_bb, f"{f}.json"), folder_bb_valid)

    if args.list_valid:  # output a list of valid frames
        with open(os.path.join(folder, "valid_bboxes.txt"), "w+") as f:
            f.writelines("\n".join(valid))
