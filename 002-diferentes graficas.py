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

# Create the output directory
output_dir = f'render'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'{int(time.time())}_control_panel_animation.mp4')

# Create a figure and 4x3 grid of subplots
fig, axes = plt.subplots(3, 4, figsize=(19.2, 10.8))  # 1920x1080 resolution
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# Initialize data for the plots
x = np.linspace(0, 4*np.pi, 100)
bar_data = np.random.rand(10)
scatter_data = np.cumsum(np.random.randn(2, 100), axis=1)
polar_theta = np.linspace(0, 2*np.pi, 100)
random_noise = np.random.randn(100)

# Create initial plots
lines = []
for i, ax in enumerate(axes.flat):
    if i == 0:
        line, = ax.plot(x, np.sin(x))  # Sine wave
    elif i == 1:
        line, = ax.plot(x, np.cos(x))  # Cosine wave
    elif i == 2:
        line, = ax.plot(scatter_data[0], scatter_data[1], 'bo')  # Scatter plot
    elif i == 3:
        line = ax.bar(np.arange(10), bar_data)  # Bar chart
    elif i == 4:
        line = ax.hist(random_noise, bins=20, range=(-3, 3), color='purple', alpha=0.7)  # Histogram
    elif i == 5:
        line = ax.pie([10, 20, 30, 40], startangle=0, autopct='%1.1f%%')  # Pie chart
    elif i == 6:
        line, = ax.plot(polar_theta, np.sin(2 * polar_theta))  # Polar plot
    elif i == 7:
        line, = ax.plot(x, np.exp(-x))  # Exponential decay
    elif i == 8:
        line, = ax.plot(x, np.log(x + 1))  # Logarithmic curve
    elif i == 9:
        line, = ax.plot(np.sin(x), np.cos(x))  # Spiral
    elif i == 10:
        line, = ax.plot(x, x**2)  # Bouncing ball (Quadratic)
    elif i == 11:
        line, = ax.plot(x, random_noise)  # Random noise
    lines.append(line)

# Update function for animation
def update(frame):
    global bar_data  # Declare bar_data as global so it can be modified inside this function

    for i, ax in enumerate(axes.flat):
        if i == 0:
            lines[i].set_ydata(np.sin(x + frame * 0.1))  # Update sine wave
        elif i == 1:
            lines[i].set_ydata(np.cos(x + frame * 0.1))  # Update cosine wave
        elif i == 2:
            scatter_data[0] += np.random.randn(100) * 0.01
            scatter_data[1] += np.random.randn(100) * 0.01
            ax.clear()
            ax.plot(scatter_data[0], scatter_data[1], 'bo')  # Update scatter plot
        elif i == 3:
            bar_data = np.roll(bar_data, shift=1)
            for bar, height in zip(lines[i], bar_data):
                bar.set_height(height)  # Update bar chart
        elif i == 4:
            random_noise = np.random.randn(100)
            ax.clear()
            ax.hist(random_noise, bins=20, range=(-3, 3), color='purple', alpha=0.7)  # Update histogram
        elif i == 5:
            ax.clear()
            ax.pie([10, 20, 30, 40], startangle=frame, autopct='%1.1f%%')  # Update pie chart
        elif i == 6:
            lines[i].set_ydata(np.sin(2 * polar_theta + frame * 0.1))  # Update polar plot
        elif i == 7:
            lines[i].set_ydata(np.exp(-x + frame * 0.01))  # Update exponential decay
        elif i == 8:
            lines[i].set_ydata(np.log(x + 1 + frame * 0.01))  # Update logarithmic curve
        elif i == 9:
            ax.clear()
            ax.plot(np.sin(x + frame * 0.1), np.cos(x + frame * 0.1))  # Update spiral
        elif i == 10:
            ax.clear()
            ax.plot(x, (x + frame * 0.01)**2)  # Update quadratic
        elif i == 11:
            random_noise = np.random.randn(100)
            lines[i].set_ydata(random_noise)  # Update random noise

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
