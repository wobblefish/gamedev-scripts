#!/usr/bin/env python3
"""
remove_bg_from_video.py - Automatically remove backgrounds from videos (MP4) or GIFs

Example commands:
    # Process a specific video file:
    python remove_bg_from_video.py input_video.mp4
    
    # Process a specific GIF file:
    python remove_bg_from_video.py animation.gif
    
    # Process all MP4 files in current directory (default behavior):
    python remove_bg_from_video.py

Output:
    - Creates a directory named [filename]_frames containing extracted frames
    - Creates a directory named [filename]_pngs containing processed frames with backgrounds removed
    - Each frame is saved as a transparent PNG with the background removed

Requirements:
    - rembg (pip install rembg)
    - opencv-python (pip install opencv-python)
    - numpy (pip install numpy)
    - Pillow (pip install Pillow)
    - scikit-image (pip install scikit-image)
"""

import os
import sys
import numpy as np
from rembg import remove
from PIL import Image
import cv2
from skimage import morphology

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

def remove_bg_from_frames(frames_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    for frame_file in frame_files:
        input_path = os.path.join(frames_dir, frame_file)
        output_path = os.path.join(output_dir, frame_file)
        
        # Step 1: Use rembg for initial background removal
        input_image = Image.open(input_path)
        initial_output = remove(input_image)
        
        # Step 2: Convert to numpy arrays for CV processing
        initial_output_np = np.array(initial_output)
        alpha_channel = initial_output_np[:, :, 3]
        
        # Step 3: Find the largest contour (main object)
        # Create a binary mask from alpha channel
        _, binary_mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours found, use the original output
        if not contours:
            initial_output.save(output_path)
            continue
            
        # Find the largest contour (assuming it's the main object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Step 4: Create a filled mask from the largest contour
        filled_mask = np.zeros_like(binary_mask)
        cv2.drawContours(filled_mask, [largest_contour], 0, 255, -1)  # -1 means filled
        
        # Step 5: Clean up the mask (close small holes)
        kernel = np.ones((5, 5), np.uint8)
        filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)
        
        # Optional: Remove small holes completely
        filled_mask = morphology.remove_small_holes(
            filled_mask.astype(bool), area_threshold=500
        ).astype(np.uint8) * 255
        
        # Step 6: Create the final output image
        result = initial_output_np.copy()
        # Keep original RGB values but update alpha channel
        result[:, :, 3] = filled_mask
        
        # Step 7: Save the result
        output_image = Image.fromarray(result)
        output_image.save(output_path)
        
    print(f"Processed {len(frame_files)} frames. Output in {output_dir}")

def process_video(input_path):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    frames_dir = os.path.join(os.getcwd(), f"{base_name}_frames")
    output_dir = os.path.join(os.getcwd(), f"{base_name}_pngs")
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
    remove_bg_from_frames(frames_dir, output_dir)

def process_all_mp4s():
    files = [f for f in os.listdir(os.getcwd()) if f.lower().endswith('.mp4')]
    if not files:
        print("No MP4 files found in the current directory.")
        return
    for f in files:
        process_video(f)

def main():
    if len(sys.argv) == 1:
        process_all_mp4s()
    else:
        input_path = sys.argv[1]
        if not os.path.isfile(input_path):
            print(f"File not found: {input_path}")
            return
        process_video(input_path)

if __name__ == "__main__":
    main()
