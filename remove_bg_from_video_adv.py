#!/usr/bin/env python3
"""
remove_bg_from_video_adv.py - Unified background removal and resizing for animated character videos/GIFs

Features:
    - Supports MP4, GIF, and WEBM input
    - Removes backgrounds (optimized for black backgrounds, rembg fallback)
    - Resizes output PNGs by width or height, with selectable filter
    - Progress bars for all major steps
    - Cleans up all temporary files/folders automatically

Usage Example:
    python remove_bg_from_video_adv.py --black-bg --threshold 50 --fps 25 --height 300 --filter lanczos your_animation.mp4

Required Packages:
    pip install rembg opencv-python numpy Pillow scikit-image tqdm

CLI Options:
    --black-bg         Force black background detection mode
    --threshold VALUE  Black bg detection threshold (default: 30)
    --no-rembg         Disable rembg fallback for non-black backgrounds
    --fps VALUE        Target frames per second (default: original FPS)
    --width VALUE      Target width in pixels (mutually exclusive with --height)
    --height VALUE     Target height in pixels (mutually exclusive with --width)
    --filter NAME      Resize filter: nearest, box, bilinear, hamming, bicubic, lanczos (default: lanczos)
    input_file         Input MP4, GIF, or WEBM file

Example Usage:
    python remove_bg_from_video_adv.py --black-bg --threshold 50 --fps 25 --height 300 --filter bicubic your_animation.mp4
    python remove_bg_from_video_adv.py --black-bg --threshold 50 --fps 25 --height 300 your_animation.mp4
    python remove_bg_from_video_adv.py your_animation.mp4

All output PNGs will be written to a single folder. Temporary frames are deleted after processing.
"""

import os
import sys
import argparse
import shutil
import numpy as np
from rembg import remove
from PIL import Image
import cv2
from skimage import morphology
from tqdm import tqdm

RESAMPLE_FILTERS = {
    'nearest': Image.Resampling.NEAREST,
    'box': Image.Resampling.BOX,
    'bilinear': Image.Resampling.BILINEAR,
    'hamming': Image.Resampling.HAMMING,
    'bicubic': Image.Resampling.BICUBIC,
    'lanczos': Image.Resampling.LANCZOS
}

def extract_frames(input_path, frames_dir, fps=None):
    os.makedirs(frames_dir, exist_ok=True)
    if input_path.lower().endswith(('.mp4', '.webm')):
        vidcap = cv2.VideoCapture(input_path)
        original_fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_interval = 1
        if fps and fps < original_fps:
            frame_interval = int(round(original_fps / fps))
        count = 0
        frame_number = 0
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total_frames, desc="Extracting frames") as pbar:
            while True:
                success, image = vidcap.read()
                if not success:
                    break
                if frame_number % frame_interval == 0:
                    frame_path = os.path.join(frames_dir, f"frame_{count:04d}.png")
                    cv2.imwrite(frame_path, image)
                    count += 1
                frame_number += 1
                pbar.update(1)
        vidcap.release()
        return count
    elif input_path.lower().endswith('.gif'):
        img = Image.open(input_path)
        count = 0
        try:
            with tqdm(desc="Extracting frames") as pbar:
                while True:
                    img.seek(count)
                    frame = img.convert('RGBA')
                    frame_path = os.path.join(frames_dir, f"frame_{count:04d}.png")
                    frame.save(frame_path)
                    count += 1
                    pbar.update(1)
        except EOFError:
            pass
        return count
    else:
        raise ValueError("Unsupported file type. Only MP4, WEBM, and GIF are supported.")

def remove_bg_from_frames(frames_dir, output_dir, black_bg_threshold=30, use_rembg=True, force_black_bg=False):
    os.makedirs(output_dir, exist_ok=True)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    for frame_file in tqdm(frame_files, desc="Removing BG"):
        input_path = os.path.join(frames_dir, frame_file)
        output_path = os.path.join(output_dir, frame_file)
        input_image = Image.open(input_path)
        input_np = np.array(input_image)
        # If force_black_bg, always use black bg method
        if force_black_bg:
            if len(input_np.shape) == 3 and input_np.shape[2] >= 3:
                if input_np.shape[2] == 3:
                    input_hsv = cv2.cvtColor(input_np, cv2.COLOR_RGB2HSV)
                else:
                    input_hsv = cv2.cvtColor(input_np[:,:,:3], cv2.COLOR_RGB2HSV)
                if input_np.shape[2] == 3:
                    rgba = cv2.cvtColor(input_np, cv2.COLOR_RGB2RGBA)
                else:
                    rgba = input_np.copy()
                mask = cv2.inRange(input_hsv, np.array([0, 0, 0]), np.array([180, 255, black_bg_threshold]))
                mask = 255 - mask
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                rgba[:, :, 3] = mask
                output_image = Image.fromarray(rgba)
                output_image.save(output_path)
            else:
                input_image.save(output_path)
            continue
        # Otherwise, auto-detect
        has_black_bg = False
        if len(input_np.shape) == 3 and input_np.shape[2] >= 3:
            if input_np.shape[2] == 3:
                input_hsv = cv2.cvtColor(input_np, cv2.COLOR_RGB2HSV)
            else:
                input_hsv = cv2.cvtColor(input_np[:,:,:3], cv2.COLOR_RGB2HSV)
            h, w = input_hsv.shape[:2]
            border_pixels = np.concatenate([
                input_hsv[0, :],
                input_hsv[-1, :],
                input_hsv[:, 0],
                input_hsv[:, -1]
            ])
            dark_pixels = np.sum(border_pixels[:, 2] < black_bg_threshold)
            if dark_pixels / len(border_pixels) > 0.8:
                has_black_bg = True
        if has_black_bg:
            if input_np.shape[2] == 3:
                rgba = cv2.cvtColor(input_np, cv2.COLOR_RGB2RGBA)
            else:
                rgba = input_np.copy()
            mask = cv2.inRange(input_hsv, np.array([0, 0, 0]), np.array([180, 255, black_bg_threshold]))
            mask = 255 - mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            rgba[:, :, 3] = mask
            output_image = Image.fromarray(rgba)
            output_image.save(output_path)
        elif use_rembg:
            initial_output = remove(input_image)
            initial_output_np = np.array(initial_output)
            alpha_channel = initial_output_np[:, :, 3]
            _, binary_mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
            binary_mask = binary_mask.astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                initial_output.save(output_path)
                continue
            largest_contour = max(contours, key=cv2.contourArea)
            filled_mask = np.zeros_like(binary_mask)
            cv2.drawContours(filled_mask, [largest_contour], 0, 255, -1)
            kernel = np.ones((5, 5), np.uint8)
            filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)
            filled_mask = morphology.remove_small_holes(
                filled_mask.astype(bool), area_threshold=500
            ).astype(np.uint8) * 255
            result = initial_output_np.copy()
            result[:, :, 3] = filled_mask
            output_image = Image.fromarray(result)
            output_image.save(output_path)
        else:
            input_image.save(output_path)

def resize_pngs(input_dir, output_dir, target_width=None, target_height=None, filter_name='lanczos'):
    os.makedirs(output_dir, exist_ok=True)
    resampling_filter = RESAMPLE_FILTERS.get(filter_name.lower(), Image.Resampling.LANCZOS)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    for img_file in tqdm(image_files, desc="Resizing"):
        in_path = os.path.join(input_dir, img_file)
        out_path = os.path.join(output_dir, img_file)
        with Image.open(in_path) as img:
            orig_width, orig_height = img.size
            if target_width is not None:
                width_percent = (target_width / float(orig_width))
                new_height = int((float(orig_height) * float(width_percent)))
                new_width = target_width
            elif target_height is not None:
                height_percent = (target_height / float(orig_height))
                new_width = int((float(orig_width) * float(height_percent)))
                new_height = target_height
            else:
                raise ValueError("Either target_width or target_height must be specified")
            resized_img = img.resize((new_width, new_height), resampling_filter)
            resized_img.save(out_path, format='PNG', optimize=True)

def main():
    parser = argparse.ArgumentParser(description='Remove background and resize frames from MP4, GIF, or WEBM')
    parser.add_argument('--black-bg', action='store_true', help='Force black background detection mode')
    parser.add_argument('--threshold', type=int, default=30, help='Black bg detection threshold (default: 30)')
    parser.add_argument('--no-rembg', action='store_true', help='Disable rembg fallback for non-black backgrounds')
    parser.add_argument('--fps', type=float, help='Target frames per second (default: original FPS)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--width', type=int, help='Target width in pixels (maintain aspect ratio)')
    group.add_argument('--height', type=int, help='Target height in pixels (maintain aspect ratio)')
    parser.add_argument('--filter', default='lanczos', choices=list(RESAMPLE_FILTERS.keys()), help='Resize filter (default: lanczos)')
    parser.add_argument('input_file', nargs=1, help='Input MP4, GIF, or WEBM file')
    args = parser.parse_args()

    input_path = args.input_file[0]
    if not os.path.isfile(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    temp_frames_dir = os.path.join(os.getcwd(), f"_{base_name}_frames_tmp")
    temp_bg_dir = os.path.join(os.getcwd(), f"_{base_name}_bg_tmp")
    output_dir = os.path.join(os.getcwd(), f"{base_name}_pngs")
    try:
        print("\n[Step 1/3] Extracting frames...")
        extract_frames(input_path, temp_frames_dir, fps=args.fps)
        print("[Step 2/3] Removing background...")
        remove_bg_from_frames(
            temp_frames_dir,
            temp_bg_dir,
            black_bg_threshold=args.threshold,
            use_rembg=not args.no_rembg,
            force_black_bg=args.black_bg
        )
        print("[Step 3/3] Resizing and saving PNGs...")
        resize_pngs(
            temp_bg_dir,
            output_dir,
            target_width=args.width,
            target_height=args.height,
            filter_name=args.filter
        )
        print(f"\nDone! Output PNGs are in: {output_dir}\n")
    finally:
        # Clean up temp folders
        for d in [temp_frames_dir, temp_bg_dir]:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)

if __name__ == "__main__":
    main()

def extract_frames_from_mp4(mp4_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(mp4_path)
    count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        frame_path = os.path.join(frames_dir, f"frame_{count:04d}.png")
        cv2.imwrite(frame_path, image)
        count += 1
    vidcap.release()
    return count

def extract_frames_from_gif(gif_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    img = Image.open(gif_path)
    count = 0
    try:
        while True:
            img.seek(count)
            frame = img.convert('RGBA')
            frame_path = os.path.join(frames_dir, f"frame_{count:04d}.png")
            frame.save(frame_path)
            count += 1
    except EOFError:
        pass
    return count

def remove_black_shadows(image_np, alpha_np, shadow_thresh=60, shadow_alpha_thresh=180):
    # Remove black/dark pixels near the alpha edge
    # image_np: HxWx4 RGBA, alpha_np: HxW
    rgb = image_np[..., :3]
    mask = (alpha_np > shadow_alpha_thresh)
    # Convert to HSV for better shadow detection
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Shadows: low V (brightness) and low S (saturation)
    shadow_mask = (hsv[..., 2] < shadow_thresh) & mask
    # Set alpha to 0 for detected shadow pixels
    new_alpha = np.copy(alpha_np)
    new_alpha[shadow_mask] = 0
    return new_alpha

def remove_bg_from_frames_adv(frames_dir, output_dir, alpha_thresh=8, shadow_thresh=60, shadow_alpha_thresh=180, min_area=300):
    os.makedirs(output_dir, exist_ok=True)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    for frame_file in frame_files:
        input_path = os.path.join(frames_dir, frame_file)
        output_path = os.path.join(output_dir, frame_file)
        input_image = Image.open(input_path)
        initial_output = remove(input_image)
        initial_output_np = np.array(initial_output)
        alpha_channel = initial_output_np[:, :, 3]
        # Step 1: Lower alpha threshold to keep flashes
        _, binary_mask = cv2.threshold(alpha_channel, alpha_thresh, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)
        # Step 2: Largest contour for main object
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            initial_output.save(output_path)
            continue
        largest_contour = max(contours, key=cv2.contourArea)
        filled_mask = np.zeros_like(binary_mask)
        cv2.drawContours(filled_mask, [largest_contour], 0, 255, -1)
        # Step 3: Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)
        filled_mask = morphology.remove_small_holes(
            filled_mask.astype(bool), area_threshold=min_area
        ).astype(np.uint8) * 255
        # Step 4: Remove black shadows/outline
        new_alpha = remove_black_shadows(initial_output_np, filled_mask, shadow_thresh, shadow_alpha_thresh)
        # Step 5: Compose final output
        result = initial_output_np.copy()
        result[:, :, 3] = new_alpha
        output_image = Image.fromarray(result)
        output_image.save(output_path)
    print(f"Processed {len(frame_files)} frames. Output in {output_dir}")

def process_video(input_path, **kwargs):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    frames_dir = os.path.join(os.getcwd(), f"{base_name}_frames")
    output_dir = os.path.join(os.getcwd(), f"{base_name}_pngs_adv")
    if input_path.lower().endswith('.mp4'):
        print(f"Extracting frames from MP4: {input_path}")
        extract_frames_from_mp4(input_path, frames_dir)
    elif input_path.lower().endswith('.gif'):
        print(f"Extracting frames from GIF: {input_path}")
        extract_frames_from_gif(input_path, frames_dir)
    else:
        print("Unsupported file type. Only MP4 and GIF are supported.")
        return
    print(f"Removing background from frames in {frames_dir}")
    remove_bg_from_frames_adv(frames_dir, output_dir, **kwargs)

def process_all_mp4s(**kwargs):
    files = [f for f in os.listdir(os.getcwd()) if f.lower().endswith('.mp4')]
    if not files:
        print("No MP4 files found in the current directory.")
        return
    for f in files:
        process_video(f, **kwargs)

def main():
    # Allow parameter tweaking via environment variables (optional)
    alpha_thresh = int(os.getenv('BGREM_ALPHA_THRESH', '8'))
    shadow_thresh = int(os.getenv('BGREM_SHADOW_THRESH', '60'))
    shadow_alpha_thresh = int(os.getenv('BGREM_SHADOW_ALPHA_THRESH', '180'))
    min_area = int(os.getenv('BGREM_MIN_AREA', '300'))
    kwargs = dict(
        alpha_thresh=alpha_thresh,
        shadow_thresh=shadow_thresh,
        shadow_alpha_thresh=shadow_alpha_thresh,
        min_area=min_area
    )
    if len(sys.argv) == 1:
        process_all_mp4s(**kwargs)
    else:
        input_path = sys.argv[1]
        if not os.path.isfile(input_path):
            print(f"File not found: {input_path}")
            return
        process_video(input_path, **kwargs)

if __name__ == "__main__":
    main()
