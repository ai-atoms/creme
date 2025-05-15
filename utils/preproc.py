'''
python3
creme utils.preproc module
utility functions for extracting frames from videos
uses HyperSpy to manage dm4 files;
'''

import os
import time

import cv2
import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np


def integrate_frames_gatan(target_folder, output_file, n_frames=8, clahe=False, plot=False):
    # Function description
    image_files = [f for f in os.listdir(target_folder) if f.endswith(('.dm4', '.tiff'))]
    # Skip excessive frames
    if n_frames >= 0:
        image_files = image_files[:n_frames]
        print(f'Working with {len(image_files)} frames...')
    # Load all images into a list of HyperSpy signals
    i = 0
    signals = []
    for file in image_files:
        signal = hs.load(os.path.join(target_folder, file))
        data = signal.data.astype(np.float32)
        signal2 = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
        result_signal2 = hs.signals.Signal2D(signal2)
        result_signal2.change_dtype("uint8") # now it works
        i += 1
        signals.append(result_signal2)

    # Check if there are any signals to add
    if len(signals) == 0:
        print("No images found in the specified folder.")
    else:
        # Initialize the sum with the first signal
        sum_signal = signals[0].data.astype(np.float32)  # Use float32 to prevent overflow

        # Iterate over the remaining signals and add them
        for signal in signals[1:]:
            sum_signal += signal.data.astype(np.float32)

        # Re-scale to 0-255 range to maximize dynamic range
        sum_signal /= len(signals)

        # Remove outliers based on the percentiles
        # Calculate the lower and upper bounds 
        lower_bound = np.percentile(sum_signal, (100 - 99.73))
        upper_bound = np.percentile(sum_signal, 99.73)

        # Clip pixel values to the defined percentile range
        clipped_signal = np.clip(sum_signal, lower_bound, upper_bound)

        # Stretching intensity values to use the full range (0 to 255)
        sum_signal = cv2.normalize(clipped_signal, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Optionally apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if clahe:
            clahef = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            sum_signal = clahef.apply(sum_signal)

        # Create a new HyperSpy signal with the summed data
        result_signal = hs.signals.Signal2D(sum_signal)
        result_signal.change_dtype("uint8")
        if plot:
            result_signal.plot()
            plt.show()

        # Export the resulting signal to a file
        result_signal.save(output_file)
        print(f"Resulting signal saved as {output_file}")


def process_hour_gatan(target_folder, output_path, delta=1, clahe=False):
    # will import a single dm4 file from each delta (s) folder and convert to tiff
    
    # Minute level
    for minute_folder in sorted(os.listdir(target_folder)):
        minute_path = os.path.join(target_folder, minute_folder)
        m = minute_folder.split('_')[1]        
        print (f'Gathering data from minute {m}')
        # Ensure it's a directory
        if not os.path.isdir(minute_path):
            continue
        
        # Second level
        for second_folder in sorted(os.listdir(minute_path)):
            second_path = os.path.join(minute_path, second_folder)
            s = second_folder.split('_')[1]            
            # Operate every (delta)s
            if int(s) % delta != 0:
                continue            
            # Ensure it's a directory
            if not os.path.isdir(second_path):
                continue
            print (f'Gathering data from second {s}')

            output_file = f'{output_path}{m}m_{s}s.tif'

            # Export the resulting signal to a file
            print (second_path, output_file)
            integrate_frames_gatan(second_path, output_file, clahe=clahe)


def preproc_hssignal(signal):
    data = signal.data.astype(np.float32)
    signal2 = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    result_signal2 = hs.signals.Signal2D(signal2)
    result_signal2.change_dtype("uint8") # now it works
    return result_signal2


def export_accelerated_video(root_path, fps=14, output_path='accelerated_video.mp4'):
    """
    Creates an accelerated video from images stored in a nested folder structure.

    Args:
        root_path (str): Path to the root directory containing images in `Minute_XX/Second_XX` structure.
        fps (int): Frames per second for the output video. Default is 14.
        output_path (str): Path to save the generated video. Default is 'accelerated_video.mp4'.

    Raises:
        ValueError: If no images are found in the specified path.
    """
    # Get all minute folders sorted
    minute_folders = sorted(
        [os.path.join(root_path, folder) for folder in os.listdir(root_path) if folder.startswith("Minute")]
    )
    # -- debug
    print (minute_folders)
    
    # Collect all images in the correct order
    image_files = []

    for minute_folder in minute_folders:
        second_folders = sorted(
            [os.path.join(minute_folder, folder) for folder in os.listdir(minute_folder) if folder.startswith("Second")]
        )
        for second_folder in second_folders:
            images = sorted(
                [os.path.join(second_folder, img) for img in os.listdir(second_folder)]
            )
            image_files.extend(images)

    # Ensure there are images
    if not image_files:
        raise ValueError("No images found in the specified path.")

    # Load the first image using hspy to get video dimensions
    signal = hs.load(image_files[0])
    frame = preproc_hssignal(signal)
    frame_data = frame.data  # Extract the image data
    height, width = frame_data.shape[:2]

    # Convert grayscale to BGR for consistency with VideoWriter
    if len(frame_data.shape) == 2:  # Grayscale
        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2BGR)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write images to video
    for image_file in image_files:
        signal = hs.load(image_file)
        frame = preproc_hssignal(signal)
        frame_data = frame.data
        if len(frame_data.shape) == 2:  # Grayscale
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2BGR)
        video.write(frame_data)

    # Release the VideoWriter object
    video.release()
    cv2.destroyAllWindows()

    print(f"Video saved as {output_path}")


def crop_video_time(input_video_path, output_video_path, time_limit=60):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    duration = total_frames / fps  # Duration in seconds

    # Calculate the number of frames to crop
    crop_frame_count = int(min(time_limit, duration) * fps)

    # Get frame width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Cropping video to {min(time_limit, duration)} seconds...")

    # Read and write frames up to the crop limit
    for frame_idx in range(crop_frame_count):
        ret, frame = cap.read()
        if not ret:
            break  # Stop if there are no more frames
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Cropped video saved as {output_video_path}")


def accelerate_mp4(input_path, output_path):
    # Open the input video
    cap = cv2.VideoCapture(input_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open the video.")
        exit()

    # Get original video properties
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Original frame rate
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video

    # Open the output video writer
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            break  # Break the loop if there are no more frames
        
        # Write every second frame to achieve 2x speed
        if frame_count % 2 == 0:
            out.write(frame)

        frame_count += 1

    # Release resources
    cap.release()
    out.release()

    print(f"Accelerated video saved as {output_path}")


def extract_frames_mp4(input_path, output_path):
    '''
    Function written with aid of ChatGPT
    Extracts one frame per second from a given .mp4 video and saves them as images.

    Parameters:
        input_path (str): Path to the input .mp4 video.
        output_path (str): Directory to save the extracted frames.
    '''
    # Create the output directory if it doesn't exist
    frames_dir = os.path.join(output_path, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Open the input video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps  # Total duration in seconds

    print(f"Video FPS: {fps}, Total Duration: {duration}s")

    # Extract one frame per second
    current_second = 0
    while True:
        # Calculate the frame number corresponding to the current second
        frame_number = int(current_second * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Move to the frame

        ret, frame = cap.read()  # Read the frame
        if not ret:
            break  # Break if no frame is returned (end of video)

        # Save the frame as an image
        frame_filename = os.path.join(frames_dir, f"{current_second + 1}.png")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved frame at {current_second}s as {frame_filename}")

        current_second += 1
        if current_second >= duration:
            break

    # Release the video capture object
    cap.release()
    print(f"Frames saved in directory: {frames_dir}")


def crop_center_video(input_path, output_path, size=512):
    '''
    Function written with aid of ChatGPT
    Crop the center of a video to a specified size and save the resulting video.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the cropped output video file.
        size (int): The size of the square crop (default is 512).
    '''
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    # Get original video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ensure the crop size fits within the video dimensions
    if size > frame_width or size > frame_height:
        raise ValueError("Crop size cannot be larger than the video dimensions.")

    # Calculate crop coordinates (centered)
    x_start = (frame_width - size) // 2
    y_start = (frame_height - size) // 2

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change codec as needed
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (size, size))

    # Process each frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame to the center square
        cropped_frame = frame[y_start:y_start+size, x_start:x_start+size]

        # Write the cropped frame to the output video
        out.write(cropped_frame)

        frame_count += 1

    # Release resources
    cap.release()
    out.release()

    print(f"Processed {frame_count}/{total_frames} frames. Output saved to {output_path}.")


if __name__ == '__main__':
    print ('[creme] utils.preproc module loaded;')

