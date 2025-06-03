#!/usr/bin/env python3
"""
remove_bg_from_video.py - Automatically remove backgrounds from videos (MP4, WebM) or GIFs

WebM Support - Now handles WebM, MP4, and GIF files equally well

Black Background Optimization - Includes a specialized algorithm for videos with black backgrounds:

Automatically detects if a frame has a dark background

Uses a fast color-based threshold approach instead of the heavier rembg model

Preserves all non-black pixels while making black pixels transparent

COMMAND LINE OPTIONS FOR FINE-TUNING:
    --black-bg - Forces black background detection mode
    --threshold VALUE - Adjusts how dark pixels need to be to count as "black" (default: 30)
    --no-rembg - Disables the rembg fallback for non-black backgrounds
    --fps VALUE - Sets the target frames per second (default: original FPS)

EXAMPLE USAGE:
# Process a WebM file with black background
python remove_bg_from_video.py your_animation.webm

# Process with a more aggressive threshold (if some darker gray areas aren't being removed)
python remove_bg_from_video.py --black-bg --threshold 50 your_animation.webm

# Process all supported videos in the current directory
python remove_bg_from_video.py

The script will automatically detect dark backgrounds, but you can use the --black-bg flag to force this mode. If you find that some parts of the background aren't being removed properly, try adjusting the threshold value.

Example commands:
    # Process a specific video file:
    python remove_bg_from_video.py input_video.mp4
    
    # Process a WebM file:
    python remove_bg_from_video.py animation.webm
    
    # Process a specific GIF file:
    python remove_bg_from_video.py animation.gif
    
    # Process all video files in current directory (default behavior):
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

def extract_frames_from_video(video_path, frames_dir, fps=None):
    """Extract frames from a video file (MP4, WebM, etc.)
    
    Args:
        video_path: Path to the video file
        frames_dir: Directory to save extracted frames
        fps: Target frames per second (None for original FPS)
    """
    os.makedirs(frames_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    
    # Get original video FPS and calculate frame interval
    original_fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1
    if fps and fps < original_fps:
        frame_interval = int(round(original_fps / fps))
    
    count = 0
    frame_number = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
            
        # Only process frames at the specified interval
        if frame_number % frame_interval == 0:
            frame_path = os.path.join(frames_dir, f"frame_{count:04d}.png")
            cv2.imwrite(frame_path, image)
            count += 1
            
        frame_number += 1
        
    vidcap.release()
    print(f"Extracted {count} frames at {fps if fps else original_fps:.1f} FPS")
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

def remove_bg_from_frames(frames_dir, output_dir, black_bg_threshold=30, use_rembg=True):
    """Remove backgrounds from frames with optimization for black backgrounds"""
    os.makedirs(output_dir, exist_ok=True)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    for frame_file in frame_files:
        input_path = os.path.join(frames_dir, frame_file)
        output_path = os.path.join(output_dir, frame_file)
        
        # Load the image
        input_image = Image.open(input_path)
        input_np = np.array(input_image)
        
        # Method selection: check if the image has a black background
        # If it does, use color-based removal for better performance
        has_black_bg = False
        
        if len(input_np.shape) == 3 and input_np.shape[2] >= 3:
            # Convert to HSV for better color analysis
            if input_np.shape[2] == 3:
                input_hsv = cv2.cvtColor(input_np, cv2.COLOR_RGB2HSV)
            else:  # RGBA
                input_hsv = cv2.cvtColor(input_np[:,:,:3], cv2.COLOR_RGB2HSV)
                
            # Check if a significant portion of the border pixels are black/very dark
            h, w = input_hsv.shape[:2]
            border_pixels = np.concatenate([
                input_hsv[0, :],       # top row
                input_hsv[-1, :],      # bottom row
                input_hsv[:, 0],       # left column
                input_hsv[:, -1]       # right column
            ])
            dark_pixels = np.sum(border_pixels[:, 2] < black_bg_threshold)  # V channel < threshold
            if dark_pixels / len(border_pixels) > 0.8:  # If >80% of border is dark
                has_black_bg = True
        
        # Process based on background type
        if has_black_bg:
            # For black backgrounds: use simple color threshold
            if len(input_np.shape) == 3 and input_np.shape[2] >= 3:
                # Create mask where black/very dark pixels become transparent
                if input_np.shape[2] == 3:
                    # Convert BGR to BGRA
                    rgba = cv2.cvtColor(input_np, cv2.COLOR_RGB2RGBA)
                else:
                    rgba = input_np.copy()
                
                # Create a mask from dark pixels
                mask = cv2.inRange(input_hsv, np.array([0, 0, 0]), np.array([180, 255, black_bg_threshold]))
                # Invert: 0 for black background, 255 for content
                mask = 255 - mask
                
                # Clean up the mask
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                # Apply the mask to the alpha channel
                rgba[:, :, 3] = mask
                
                # Save the result
                output_image = Image.fromarray(rgba)
                output_image.save(output_path)
            else:
                # If not RGB/RGBA, fall back to original image
                input_image.save(output_path)
        elif use_rembg:
            # For other backgrounds: use rembg
            initial_output = remove(input_image)
            
            # Convert to numpy arrays for CV processing
            initial_output_np = np.array(initial_output)
            alpha_channel = initial_output_np[:, :, 3]
            
            # Find the largest contour (main object)
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
            
            # Create a filled mask from the largest contour
            filled_mask = np.zeros_like(binary_mask)
            cv2.drawContours(filled_mask, [largest_contour], 0, 255, -1)  # -1 means filled
            
            # Clean up the mask (close small holes)
            kernel = np.ones((5, 5), np.uint8)
            filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)
            
            # Remove small holes completely
            filled_mask = morphology.remove_small_holes(
                filled_mask.astype(bool), area_threshold=500
            ).astype(np.uint8) * 255
            
            # Create the final output image
            result = initial_output_np.copy()
            # Keep original RGB values but update alpha channel
            result[:, :, 3] = filled_mask
            
            # Save the result
            output_image = Image.fromarray(result)
            output_image.save(output_path)
        else:
            # Fall back to just using simple transparency
            input_image.save(output_path)
        
    print(f"Processed {len(frame_files)} frames. Output in {output_dir}")

def process_video(input_path, black_bg_threshold=30, use_rembg=True, fps=None):
    """Process video or GIF file by extracting frames and removing backgrounds
    
    Args:
        input_path: Path to input video/GIF file
        black_bg_threshold: Threshold for black background detection (0-255)
        use_rembg: Whether to use rembg for non-black backgrounds
        fps: Target frames per second (None for original FPS)
    """
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    frames_dir = os.path.join(os.getcwd(), f"{base_name}_frames")
    output_dir = os.path.join(os.getcwd(), f"{base_name}_pngs")
    
    # Handle different file types
    if input_path.lower().endswith(('.mp4', '.webm')):
        print(f"Extracting frames from video: {input_path}")
        extract_frames_from_video(input_path, frames_dir, fps)
    elif input_path.lower().endswith('.gif'):
        print(f"Extracting frames from GIF: {input_path}")
        extract_frames_from_gif(input_path, frames_dir)
    else:
        print("Unsupported file type. Only MP4, WebM, and GIF are supported.")
        return
    
    print(f"Removing background from frames in {frames_dir}")
    remove_bg_from_frames(frames_dir, output_dir, black_bg_threshold, use_rembg)

def process_all_videos():
    """Process all video and GIF files in the current directory"""
    # Find all supported video files
    video_files = []
    for ext in ['.mp4', '.webm', '.gif']:
        video_files.extend([f for f in os.listdir(os.getcwd()) if f.lower().endswith(ext)])
    
    if not video_files:
        print("No supported video files found in the current directory.")
        return
    
    for f in video_files:
        process_video(f)

def main():
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Check for flags in command line arguments
        args = sys.argv[1:]
        input_path = args[-1]  # Assume the last argument is the file path
        
        # Process flags
        black_bg_threshold = 30  # Default threshold for black background detection
        use_rembg = True  # Default to using rembg for non-black backgrounds
        fps = None  # Default to original FPS
        
        # Process flags
        i = 0
        while i < len(args) - 1:  # Skip the last argument (input file)
            if args[i] == '--black-bg':
                pass  # Flag is handled below
            elif args[i] == '--threshold':
                try:
                    black_bg_threshold = int(args[i + 1])
                    i += 1  # Skip the next argument
                except (ValueError, IndexError):
                    print(f"Warning: Invalid threshold value after --threshold, using default: {black_bg_threshold}")
            elif args[i] == '--fps':
                try:
                    fps = float(args[i + 1])
                    if fps <= 0:
                        print("Warning: FPS must be greater than 0, using original FPS")
                        fps = None
                    i += 1  # Skip the next argument
                except (ValueError, IndexError):
                    print("Warning: Invalid FPS value after --fps, using original FPS")
            i += 1
        
        # Flag to disable rembg fallback
        if '--no-rembg' in args:
            use_rembg = False
        
        # Process the input file if it exists
        if os.path.isfile(input_path):
            process_video(input_path, black_bg_threshold, use_rembg, fps)
        else:
            print(f"File not found: {input_path}")
    else:
        # No arguments, process all videos in the current directory
        process_all_videos()

if __name__ == "__main__":
    main()
