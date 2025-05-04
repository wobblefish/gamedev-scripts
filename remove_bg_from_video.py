import os
import sys
from rembg import remove
from PIL import Image
import cv2

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
        input_image = Image.open(input_path)
        output_image = remove(input_image)
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
