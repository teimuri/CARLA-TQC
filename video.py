import cv2
import os

def create_video_from_images(image_folder, output_video, frame_rate):
    # Get list of image file paths
    images = sorted(
        [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    )
    
    if not images:
        print("No images found in the folder.")
        return

    # Get the size of the first image to set video dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    print(f"Video created successfully at {output_video}")

# Parameters
image_folder = "/media/carla/AVRL/our_ppo/image_logs/100"  # Replace with your folder path
output_video = "output_video.mp4"    # Replace with your desired output path
frame_rate = 1 / 0.1                # 2.5 FPS

create_video_from_images(image_folder, output_video, frame_rate)
