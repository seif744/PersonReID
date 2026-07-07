"""
Prepare RandPerson subset images for local ReID evaluation/training.

RandPerson filenames encode labels:
    000000_s00_c00_f000228.jpg
    person_id scene camera frame

This script keeps the original images and creates symlinks or copies in:
    prepared/randperson/train/<person_id>/*.jpg
    prepared/randperson/query/<camera>/<person_id>/*.jpg
    prepared/randperson/gallery/<camera>/<person_id>/*.jpg

Use symlinks by default to avoid duplicating the 1.4GB subset.
"""

import argparse
import os
import re
import shutil
from collections import defaultdict


PATTERN = re.compile(
    r"^(?P<pid>\d+)_s(?P<scene>\d+)_c(?P<cam>\d+)_f(?P<frame>\d+)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)


def find_images(root):
    items = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            match = PATTERN.match(filename)
            if not match:
                continue
            items.append({
                "path": os.path.abspath(os.path.join(dirpath, filename)),
                "pid": match.group("pid"),
                "camera": f"s{match.group('scene')}_c{match.group('cam')}",
                "frame": int(match.group("frame")),
                "filename": filename,
            })
    return items


def ensure_clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def link_or_copy(src, dst, copy):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def split_person_images(images, query_per_identity):
    images = sorted(images, key=lambda x: (x["camera"], x["frame"], x["filename"]))
    query = images[:query_per_identity]
    gallery = images[query_per_identity:]
    train = images
    return train, query, gallery


def write_split(items, out_root, split, copy):
    for item in items:
        if split == "train":
            rel = os.path.join(split, item["pid"], item["filename"])
        else:
            rel = os.path.join(split, item["camera"], item["pid"], item["filename"])
        link_or_copy(item["path"], os.path.join(out_root, rel), copy)


def write_manifest(items, path, split):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("path,person_id,camera,split\n")
        for item in items:
            f.write(f"{item['path']},{item['pid']},{item['camera']},{split}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="randperson_subset/randperson_subset")
    parser.add_argument("--out", default="prepared/randperson")
    parser.add_argument("--query-per-identity", type=int, default=1)
    parser.add_argument("--copy", action="store_true",
                        help="Copy images instead of creating symlinks.")
    args = parser.parse_args()

    items = find_images(args.src)
    if not items:
        raise SystemExit(f"No RandPerson images found under {args.src}")

    by_pid = defaultdict(list)
    for item in items:
        by_pid[item["pid"]].append(item)

    ensure_clean_dir(args.out)
    train_all, query_all, gallery_all = [], [], []
    for pid in sorted(by_pid):
        train, query, gallery = split_person_images(
            by_pid[pid], args.query_per_identity)
        train_all.extend(train)
        query_all.extend(query)
        gallery_all.extend(gallery)

    write_split(train_all, args.out, "train", args.copy)
    write_split(query_all, args.out, "query", args.copy)
    write_split(gallery_all, args.out, "gallery", args.copy)
    write_manifest(train_all, os.path.join(args.out, "train_manifest.csv"), "train")
    write_manifest(query_all, os.path.join(args.out, "query_manifest.csv"), "query")
    write_manifest(gallery_all, os.path.join(args.out, "gallery_manifest.csv"), "gallery")

    cameras = {x["camera"] for x in items}
    print(f"source images: {len(items)}")
    print(f"identities: {len(by_pid)}")
    print(f"cameras: {len(cameras)}")
    print(f"train images: {len(train_all)}")
    print(f"query images: {len(query_all)}")
    print(f"gallery images: {len(gallery_all)}")
    print(f"prepared: {args.out}")


if __name__ == "__main__":
    main()
