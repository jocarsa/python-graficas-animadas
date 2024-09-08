import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import os
import time
from datetime import datetime, timedelta

# Define video parameters
fps = 60
duration = 60  # 1 minute
total_frames = fps * duration
frame_size = (1920, 1080)

# Create the output directory with the current epoch time
epoch_time = str(int(time.time()))
output_dir = f'render'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, '{epoch_time}_control_panel_animation.mp4')

# Create a figure and 4x3 grid of subplots
fig, axes = plt.subplots(3, 4, figsize=(19.2, 10.8))  # 1920x1080 resolution
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# Initialize data for the plots
x = np.linspace(0, 4*np.pi, 100)
lines = []
for ax in axes.flat:
    line, = ax.plot(x, np.sin(x))
    lines.append(line)

# Update function for animation
def update(frame):
    for i, line in enumerate(lines):
        line.set_ydata(np.sin(x + frame * 0.1 + i))  # Simple wave animation
    return lines

# Initialize the video writer using OpenCV
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# Function to save each frame using OpenCV
def save_frame(frame):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    video_writer.write(img)

# Start time
start_time = time.time()

# Generate the frames and save the video
for frame in range(total_frames):
    update(frame)
    save_frame(frame)
    
    # Print statistics every 60 frames
    if frame % 60 == 0 and frame > 0:
        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / frame) * (total_frames - frame)
        estimated_finish = datetime.now() + timedelta(seconds=remaining_time)
        percentage_complete = (frame / total_frames) * 100
        
        print(f"Frame: {frame}/{total_frames}")
        print(f"Time Passed: {timedelta(seconds=int(elapsed_time))}")
        print(f"Time Remaining: {timedelta(seconds=int(remaining_time))}")
        print(f"Estimated Finish Time: {estimated_finish.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Percentage Complete: {percentage_complete:.2f}%\n")

# Release the video writer
video_writer.release()
plt.close(fig)

output_path
