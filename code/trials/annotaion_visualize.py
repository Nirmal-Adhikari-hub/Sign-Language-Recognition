import pandas as pd

# Path to the CSV file
anno_path = '/nas/Dataset/Phoenix/phoenix-2014-keypoints.csv'

# Path to save the log file
log_file_path = '/home/nirmal/sm/code/video_keypoints_log.txt'

# Specify the target video name (update this with your desired video name)
target_video = 'fullFrame-210x260px/train/01April_2010_Thursday_heute_default-0/1/01April_2010_Thursday_heute'

try:
    # Load the CSV file
    print("Loading the CSV file...")
    annotations = pd.read_csv(anno_path)
    print(f"CSV loaded successfully with {len(annotations)} rows.")

    # Filter for the target video
    print(f"Filtering rows for video: {target_video}")
    video_annotations = annotations[annotations['Video'] == target_video]

    if video_annotations.empty:
        print(f"No data found for the video: {target_video}")
    else:
        print(f"Filtered {len(video_annotations)} rows for video: {target_video}")

        # Group by frames
        grouped = video_annotations.groupby('Frame')
        print(f"Number of frames found: {len(grouped)}")

        # Open the log file for writing
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"Keypoints for Video: {target_video}\n")
            log_file.write("=" * 80 + "\n")
            
            # Process and write keypoints frame by frame
            for i, (frame, frame_data) in enumerate(grouped):
                print(f"Processing Frame {frame} ({i+1}/{len(grouped)})")
                log_file.write(f"Frame: {frame}\n")
                log_file.write(frame_data.to_string(index=False) + "\n")
                log_file.write("=" * 80 + "\n")
        
        print(f"Keypoints for video '{target_video}' have been saved to '{log_file_path}'.")

except Exception as e:
    print(f"An error occurred: {e}")
