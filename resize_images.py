#!/usr/bin/env python3
"""
resize_images.py - Resize images in a folder to a specified width or height while maintaining aspect ratio

Example commands:
    # Resize all images in folder to 500px width (preserving aspect ratio):
    python resize_images.py --input_folder ./original_images --output_folder ./resized_images --width 500
    
    # Resize all images in folder to 300px height (preserving aspect ratio):
    python resize_images.py --input_folder ./original_images --output_folder ./resized_images --height 300
    
    # Specify the resampling filter method:
    python resize_images.py --input_folder ./original_images --output_folder ./resized_images --width 500 --filter bicubic
    
    # Set JPEG quality and convert all images to PNG format:
    python resize_images.py --input_folder ./original_images --output_folder ./resized_images --width 500 --quality 90 --format png
    
    # Convert images to WebP format with 85 quality:
    python resize_images.py --input_folder ./original_images --output_folder ./resized_webp --width 800 --quality 85 --format webp

Options:
    --input_folder    Directory containing images to resize
    --output_folder   Directory where resized images will be saved
    --width           Target width in pixels (height will maintain aspect ratio)
    --height          Target height in pixels (width will maintain aspect ratio)
                      Note: Specify either --width OR --height, not both
    --filter          Resizing filter method (default: 'lanczos')
                      Options: 'nearest', 'box', 'bilinear', 'hamming', 'bicubic', 'lanczos'
    --quality         JPEG quality (1-100, default: 95)
    --format          Output format (default: same as input)
                      Options: 'jpeg', 'png', 'webp', etc.

Requirements:
    - Pillow (pip install Pillow)
"""

import os
import argparse
from PIL import Image
import concurrent.futures
from pathlib import Path

RESAMPLE_FILTERS = {
    'nearest': Image.Resampling.NEAREST,
    'box': Image.Resampling.BOX,
    'bilinear': Image.Resampling.BILINEAR,
    'hamming': Image.Resampling.HAMMING,
    'bicubic': Image.Resampling.BICUBIC,
    'lanczos': Image.Resampling.LANCZOS
}

def resize_image(image_path, output_path, target_width=None, target_height=None, resampling_filter=None, quality=95, output_format=None):
    """
    Resize a single image to target width or height while maintaining aspect ratio.
    Either target_width OR target_height must be specified, not both.
    """
    try:
        with Image.open(image_path) as img:
            # Get original dimensions
            orig_width, orig_height = img.size
            
            # Calculate new dimensions to maintain aspect ratio
            if target_width is not None:
                # Resize based on width
                width_percent = (target_width / float(orig_width))
                new_height = int((float(orig_height) * float(width_percent)))
                new_width = target_width
            elif target_height is not None:
                # Resize based on height
                height_percent = (target_height / float(orig_height))
                new_width = int((float(orig_width) * float(height_percent)))
                new_height = target_height
            else:
                raise ValueError("Either target_width or target_height must be specified")
            
            # Resize the image with the specified filter
            resized_img = img.resize((new_width, new_height), resampling_filter)
            
            # Determine output format
            if output_format:
                # Make sure we have a valid extension
                if not output_path.endswith(f'.{output_format.lower()}'):
                    output_path = os.path.splitext(output_path)[0] + f'.{output_format.lower()}'
                save_format = output_format.upper()
            else:
                # Use original format
                save_format = img.format
            
            # Save the resized image
            if save_format == 'JPEG':
                resized_img.save(output_path, quality=quality, optimize=True)
            elif save_format == 'PNG':
                resized_img.save(output_path, optimize=True)
            elif save_format == 'WEBP':
                resized_img.save(output_path, quality=quality)
            else:
                resized_img.save(output_path)
                
            return True, image_path
    except Exception as e:
        return False, f"Error processing {image_path}: {str(e)}"

def process_images(input_folder, output_folder, target_width=None, target_height=None, filter_name='lanczos', quality=95, output_format=None):
    """
    Process all images in the input folder and save resized versions to the output folder
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get the filter
    resampling_filter = RESAMPLE_FILTERS.get(filter_name.lower(), Image.Resampling.LANCZOS)
    
    # Get list of image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp')
    image_files = [f for f in Path(input_folder).glob('**/*') if f.is_file() and f.suffix.lower() in image_extensions]
    
    success_count = 0
    error_count = 0
    errors = []
    
    print(f"Found {len(image_files)} image(s) to process")
    
    # Process images in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        
        for img_path in image_files:
            # Create relative path for output
            rel_path = img_path.relative_to(input_folder)
            output_path = Path(output_folder) / rel_path
            
            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Submit to thread pool
            future = executor.submit(
                resize_image, 
                str(img_path), 
                str(output_path), 
                target_width, 
                target_height,
                resampling_filter, 
                quality,
                output_format
            )
            futures.append(future)
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            success, result = future.result()
            if success:
                success_count += 1
                if success_count % 10 == 0 or success_count == len(image_files):
                    print(f"Processed {success_count}/{len(image_files)} images")
            else:
                error_count += 1
                errors.append(result)
    
    # Report results
    print(f"\nResize Complete!")
    print(f"Successfully processed: {success_count}")
    if error_count > 0:
        print(f"Errors: {error_count}")
        for error in errors:
            print(f" - {error}")

def main():
    parser = argparse.ArgumentParser(description='Resize images in a folder to a specified width or height')
    parser.add_argument('--input_folder', required=True, help='Directory containing images to resize')
    parser.add_argument('--output_folder', required=True, help='Directory where resized images will be saved')
    
    # Create mutually exclusive group for width/height (can't specify both)
    dimension_group = parser.add_mutually_exclusive_group(required=True)
    dimension_group.add_argument('--width', type=int, help='Target width in pixels (height will maintain aspect ratio)')
    dimension_group.add_argument('--height', type=int, help='Target height in pixels (width will maintain aspect ratio)')
    
    parser.add_argument('--filter', default='lanczos', choices=list(RESAMPLE_FILTERS.keys()),
                      help='Resampling filter method')
    parser.add_argument('--quality', type=int, default=95, help='JPEG/WebP quality (1-100)')
    parser.add_argument('--format', help='Output format (jpeg, png, webp, etc.)')
    
    args = parser.parse_args()
    
    process_images(
        args.input_folder,
        args.output_folder,
        args.width,
        args.height,
        args.filter,
        args.quality,
        args.format
    )

if __name__ == "__main__":
    main()
