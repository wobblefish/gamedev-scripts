#!/usr/bin/env python3
"""
center_spritesheet.py - Center subjects in a grid-based sprite sheet with optional auto-crop or uniform-cell modes

Example commands:
    # Basic usage with specified rows and columns:
    python center_spritesheet.py --in-sheet input.png --out-sheet centered.png --rows 4 --cols 4
    
    # Specify a background color to ignore:
    python center_spritesheet.py --in-sheet input.png --out-sheet centered.png --rows 4 --cols 4 --bg-color "#FF00FF"
    
    # Auto-crop the sheet before processing:
    python center_spritesheet.py --in-sheet input.png --out-sheet centered.png --rows 4 --cols 4 --auto-crop
    
    # Create a uniform cell size based on largest content:
    python center_spritesheet.py --in-sheet input.png --out-sheet centered.png --rows 4 --cols 4 --uniform-cell
    
    # Combine options:
    python center_spritesheet.py --in-sheet input.png --out-sheet centered.png --rows 4 --cols 4 --auto-crop --uniform-cell --bg-color "#FF00FF"

Requirements:
    - Pillow (pip install Pillow)
    - numpy (pip install numpy)
"""

import argparse
import logging
from PIL import Image

import numpy as np  # no, I didn't call you a numpy LOL

def parse_args():
    p = argparse.ArgumentParser(
        description="Center subjects in a grid-based sprite sheet with optional auto-crop or uniform-cell modes."
    )
    p.add_argument("--in-sheet",    required=True, help="Path to input sprite sheet")
    p.add_argument("--out-sheet",   required=True, help="Path to write centered sheet")
    p.add_argument("--rows",        type=int,   required=True, help="Number of rows in grid")
    p.add_argument("--cols",        type=int,   required=True, help="Number of columns in grid")
    p.add_argument(
        "--bg-color",
        default=None,
        help=(
            "Background color to ignore (e.g. '#RRGGBB'). "
            "If omitted, script samples the top-left pixel."  
        ),
    )
    p.add_argument(
        "--auto-crop",
        action="store_true",
        help="Trim the overall sheet to its content bbox before slicing into cells.",
    )
    p.add_argument(
        "--uniform-cell",
        action="store_true",
        help=(
            "Rebuild sheet so each cell is the same size (max content bbox) "
            "rather than the original grid cell size."
        ),
    )
    return p.parse_args()


def hex_to_rgb(hexstr):
    hexstr = hexstr.lstrip('#')
    return tuple(int(hexstr[i:i+2], 16) for i in (0,2,4))


def detect_bg_color(image):
    """Sample top-left pixel as background if fully opaque, else return None."""
    r,g,b,a = image.getpixel((0,0))
    return (r,g,b) if a == 255 else None


def get_bbox(image, bg_color):
    """Return bbox (min_x, min_y, max_x, max_y) of all pixels != bg_color or alpha>0."""
    pixels = image.load()
    w,h = image.size
    pts = []
    if bg_color:
        for y in range(h):
            for x in range(w):
                r,g,b,a = pixels[x,y]
                if (r,g,b) != bg_color:
                    pts.append((x,y))
    else:
        for y in range(h):
            for x in range(w):
                if pixels[x,y][3] != 0:
                    pts.append((x,y))
    if not pts:
        return None
    xs, ys = zip(*pts)
    return min(xs), min(ys), max(xs), max(ys)


import numpy as np

def center_sprite(sprite, target_w, target_h, bg_color=None):
    """Center a sprite image so its median point is at the center of a transparent canvas of size target_w x target_h."""
    arr = np.array(sprite)
    # Create mask of non-background pixels
    if arr.shape[2] == 4:
        alpha = arr[:,:,3]
        mask = alpha > 0
    else:
        mask = np.ones(arr.shape[:2], dtype=bool)
    # If bg_color provided, mask those out too
    if bg_color is not None:
        bg = np.array(bg_color, dtype=np.uint8)
        rgb = arr[:,:,:3]
        mask &= ~np.all(rgb == bg, axis=2)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        # Empty sprite
        median_x = sprite.width // 2
        median_y = sprite.height // 2
    else:
        median_x = int(np.median(xs))
        median_y = int(np.median(ys))
    # Compute offset to place median at center of canvas
    offset_x = (target_w // 2) - median_x
    offset_y = (target_h // 2) - median_y
    canvas = Image.new('RGBA', (target_w, target_h), (0,0,0,0))
    canvas.paste(sprite, (offset_x, offset_y), mask=sprite)
    logging.info(f"Median at ({median_x},{median_y}), offset=({offset_x},{offset_y})")
    return canvas


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = parse_args()
    sheet = Image.open(args.in_sheet).convert('RGBA')

    logging.info(f"Input sheet: {args.in_sheet}")
    logging.info(f"Rows: {args.rows}, Cols: {args.cols}")
    # determine background
    bg = None
    if args.bg_color:
        bg = hex_to_rgb(args.bg_color)
    else:
        detected = detect_bg_color(sheet)
        if detected:
            bg = detected
            logging.info(f"Auto-detected background color: #{bg[0]:02x}{bg[1]:02x}{bg[2]:02x}")

    # auto-crop the whole sheet if requested
    if args.auto_crop:
        global_bbox = get_bbox(sheet, bg)
        if global_bbox:
            l,t,r,b = global_bbox
            sheet = sheet.crop((l, t, r+1, b+1))
            logging.info(f"Cropped sheet to content bbox: {l,t,r,b}")

    # recalc overall dimensions
    sheet_w, sheet_h = sheet.size
    cell_w = sheet_w // args.cols
    cell_h = sheet_h // args.rows

    # slice into individual frames
    frames = []
    for row in range(args.rows):
        for col in range(args.cols):
            x0 = col * cell_w
            y0 = row * cell_h
            frame = sheet.crop((x0, y0, x0+cell_w, y0+cell_h))
            frames.append(frame)
            logging.info(f"Frame ({row},{col}) cell bbox: ({x0},{y0},{x0+cell_w},{y0+cell_h})")

    # if uniform-cell, determine max content bbox size across all frames
    if args.uniform_cell:
        max_w = max_h = 0
        sprites = []
        # first, trim each frame tightly
        for idx, f in enumerate(frames):
            bbox = get_bbox(f, bg)
            if bbox:
                l,t,r,b = bbox
                sprite = f.crop((l, t, r+1, b+1))
                logging.info(f"Frame {idx} cropped bbox: {l,t,r,b}")
            else:
                sprite = Image.new('RGBA', (0,0))
                logging.info(f"Frame {idx} is empty after cropping")
            sprites.append(sprite)
            max_w = max(max_w, sprite.width)
            max_h = max(max_h, sprite.height)
        logging.info(f"Uniform cell size: {max_w}×{max_h}")
        # center each trimmed sprite in a uniform cell
        centered = []
        for idx, s in enumerate(sprites):
            centered_sprite = center_sprite(s, max_w, max_h, bg)
            centered.append(centered_sprite)
            logging.info(f"Frame {idx} centered at cell size: {max_w}x{max_h}")
        out_w = max_w * args.cols
        out_h = max_h * args.rows
        new_sheet = Image.new('RGBA', (out_w, out_h), (0,0,0,0))

        idx = 0
        for row in range(args.rows):
            for col in range(args.cols):
                x = col * max_w
                y = row * max_h
                new_sheet.paste(centered[idx], (x,y), mask=centered[idx])
                logging.info(f"Pasted frame {idx} at ({x},{y}) in output sheet")
                idx += 1

    else:
        # default centering within original cell size
        sprites = []
        for idx, f in enumerate(frames):
            bbox = get_bbox(f, bg)
            if bbox:
                l,t,r,b = bbox
                sprite = f.crop((l, t, r+1, b+1))
                logging.info(f"Frame {idx} cropped bbox: {l,t,r,b}")
            else:
                sprite = Image.new('RGBA', (0,0))
                logging.info(f"Frame {idx} is empty after cropping")
            sprites.append(sprite)
        centered = []
        for idx, s in enumerate(sprites):
            centered_sprite = center_sprite(s, cell_w, cell_h, bg)
            centered.append(centered_sprite)
            logging.info(f"Frame {idx} centered at cell size: {cell_w}x{cell_h}")
        new_sheet = Image.new('RGBA', (sheet_w, sheet_h), (0,0,0,0))

        idx = 0
        for row in range(args.rows):
            for col in range(args.cols):
                x = col * cell_w
                y = row * cell_h
                new_sheet.paste(centered[idx], (x,y), mask=centered[idx])
                logging.info(f"Pasted frame {idx} at ({x},{y}) in output sheet")
                idx += 1

    new_sheet.save(args.out_sheet)
    logging.info(f"Saved sheet to {args.out_sheet!r}")


if __name__ == '__main__':
    main()
