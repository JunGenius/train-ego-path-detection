import argparse
import json
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
import torch
from PIL import Image

from src.utils.common import simple_logger
from src.utils.interface import Detector
from src.utils.visualization import draw_egopath


def parse_arguments():
    parser = argparse.ArgumentParser(description="Ego-Path Detection Batch Inference Script with JSON ROI")
    parser.add_argument(
        "model",
        type=str,
        help="Name of the trained model to use (e.g., 'chromatic-laughter-5').",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input directory containing images and Labelme JSON files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the destination directory for the output files. If not specified, creates an 'output' folder inside the input directory.",
    )
    parser.add_argument(
        "--crop",
        type=str,
        default="auto",
        help="Fallback crop option if JSON is missing ('auto', 'none', or 'x_left,y_top,x_right,y_bottom').",
    )
    parser.add_argument(
        "--show-crop",
        action="store_true",
        help="If enabled, displays the crop boundaries in the visual output.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps"]
        + [f"cuda:{x}" for x in range(torch.cuda.device_count())],
        help="Device to use ('cpu', 'cuda', 'cuda:x' or 'mps').",
    )

    return parser.parse_args()


def parse_labelme_json(json_path):
    """解析 Labelme 的 JSON 文件，提取 ROI 区域坐标 (x_left, y_top, x_right, y_bottom)。"""
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        shapes = data.get("shapes", [])
        if not shapes:
            return None
        
        # 获取第一个标注形状的点坐标，计算其外接矩形边界
        points = shapes[0].get("points", [])
        if not points:
            return None
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        x_left = int(min(xs))
        y_top = int(min(ys))
        x_right = int(max(xs))
        y_bottom = int(max(ys))
        
        return (x_left, y_top, x_right, y_bottom)
    except Exception as e:
        print(f"Error parsing JSON {json_path}: {e}")
        return None


def main(args):
    logger = simple_logger(__name__, "info")
    base_path = os.path.dirname(__file__)

    if not os.path.isdir(args.input):
        raise NotADirectoryError(f"Input path '{args.input}' is not a valid directory.")

    # 确定输出目录
    if args.output is not None:
        output_dir = args.output
    else:
        output_dir = os.path.join(args.input, "output")
    os.makedirs(output_dir, exist_ok=True)

    # 查找输入目录下的所有图片
    image_extensions = (".jpg", ".jpeg", ".png")
    image_files = sorted([
        f for f in os.listdir(args.input) 
        if f.lower().endswith(image_extensions)
    ])

    if not image_files:
        logger.warning(f"No images found in directory: {args.input}")
        return

    logger.info(f"\nFound {len(image_files)} images in {args.input}. Starting batch inference...")

    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(args.input, img_name)
        base_name, ext = os.path.splitext(img_name)
        json_path = os.path.join(args.input, f"{base_name}.json")
        output_path = os.path.join(output_dir, f"{base_name}_out{ext}")

        # 优先从同名 JSON 中解析 crop 坐标，若无则使用 --crop 参数回退
        json_crop = parse_labelme_json(json_path)
        if json_crop is not None:
            current_crop = json_crop
        else:
            if args.crop == "auto":
                current_crop = "auto"
            elif args.crop == "none":
                current_crop = None
            else:
                current_crop = tuple(map(int, args.crop.split(",")))

        # 初始化当前图片的 Detector
        detector = Detector(
            model_path=os.path.join(base_path, "weights", args.model),
            crop_coords=current_crop,
            runtime="pytorch",
            device=args.device,
        )

        logger.info(f"[{idx+1}/{len(image_files)}] Processing {img_name} | Crop: {current_crop}")

        frame = Image.open(img_path)
        iterations = 50 if current_crop == "auto" else 1
        for _ in range(iterations):
            crop_show = detector.get_crop_coords() if args.show_crop else None
            res = detector.detect(frame)
            
        draw_egopath(frame, res, crop_coords=crop_show).save(output_path)

    logger.info(f"\nInference complete. All outputs saved to {output_dir}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)