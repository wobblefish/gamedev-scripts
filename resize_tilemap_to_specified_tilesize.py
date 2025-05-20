# resize_tilemap_to_specified_tilesize.py
# python resize_tilemap_to_specified_tilesize.py --old 70 --new 60 --image winter-sheet-70.png --top_left 0,0 --bottom_right 13,6


import argparse
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Analyze and resize tilemap image based on tile coordinates and sizes.")
    parser.add_argument('--old', type=int, required=True, help='Old tile size (e.g. 70)')
    parser.add_argument('--new', type=int, required=True, help='New tile size (e.g. 60)')
    parser.add_argument('--image', type=str, required=True, help='Path to tilemap image (e.g. sheet.png)')
    parser.add_argument('--top_left', type=str, required=True, help='Top-left tile coordinate as x,y (e.g. 0,0)')
    parser.add_argument('--bottom_right', type=str, required=True, help='Bottom-right tile coordinate as x,y (e.g. 13,6)')
    args = parser.parse_args()

    try:
        img = Image.open(args.image)
        width, height = img.size
        print(f"Tilemap image: {args.image}")
        print(f"Image size: {width} x {height} px")
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    try:
        tl_x, tl_y = map(int, args.top_left.split(','))
        br_x, br_y = map(int, args.bottom_right.split(','))
    except Exception:
        print("Invalid coordinate format. Use x,y (e.g. 0,0)")
        return

    num_tiles_x = br_x - tl_x + 1
    num_tiles_y = br_y - tl_y + 1
    print(f"Tile grid: {num_tiles_x} tiles across, {num_tiles_y} tiles down")
    print(f"Old tile size: {args.old}px, New tile size: {args.new}px")

    old_expected_width = num_tiles_x * args.old
    old_expected_height = num_tiles_y * args.old
    print(f"Expected old image size from grid: {old_expected_width} x {old_expected_height} px")

    new_width = num_tiles_x * args.new
    new_height = num_tiles_y * args.new
    print(f"If resized to new tile size: {new_width} x {new_height} px")

    # Crop to the selected tile region
    crop_left = tl_x * args.old
    crop_upper = tl_y * args.old
    crop_right = (br_x + 1) * args.old
    crop_lower = (br_y + 1) * args.old
    cropped = img.crop((crop_left, crop_upper, crop_right, crop_lower))
    print(f"Cropped region: left={crop_left}, top={crop_upper}, right={crop_right}, bottom={crop_lower}")

    # Resize to new tile size
    resized = cropped.resize((new_width, new_height), Image.NEAREST)

    # Save output
    import os
    base, ext = os.path.splitext(args.image)
    outname = f"{base}_resized-{args.new}{ext}"
    resized.save(outname)
    print(f"Saved resized tilemap as: {outname}")

if __name__ == "__main__":
    main()