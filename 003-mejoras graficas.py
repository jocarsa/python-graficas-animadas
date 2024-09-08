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
output_path = os.path.join(output_dir, f'{int(time.time())}_stylized_control_panel_animation.mp4')

# Create a figure and 4x3 grid of subplots
fig, axes = plt.subplots(3, 4, figsize=(19.2, 10.8))  # 1920x1080 resolution
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# Initialize data for the plots
x = np.linspace(0, 4*np.pi, 100)
bar_data = np.random.rand(10)
scatter_data = np.cumsum(np.random.randn(2, 100), axis=1)
polar_theta = np.linspace(0, 2*np.pi, 100)
random_noise = np.random.randn(100)

# Customizing the plots
plot_titles = [
    "Sine Wave", "Cosine Wave", "Random Walk (Scatter)", "Bar Chart",
    "Histogram", "Pie Chart", "Polar Plot", "Exponential Decay",
    "Logarithmic Curve", "Spiral", "Quadratic (Bouncing Ball)", "Random Noise"
]

for i, ax in enumerate(axes.flat):
    ax.set_title(plot_titles[i], fontsize=12, fontweight='bold')
    ax.grid(True)

# Create initial plots
lines = []
for i, ax in enumerate(axes.flat):
    if i == 0:
        line, = ax.plot(x, np.sin(x), color='blue', lw=2)  # Sine wave
    elif i == 1:
        line, = ax.plot(x, np.cos(x), color='orange', lw=2)  # Cosine wave
    elif i == 2:
        line, = ax.plot(scatter_data[0], scatter_data[1], 'go', alpha=0.7)  # Scatter plot
    elif i == 3:
        line = ax.bar(np.arange(10), bar_data, color='teal', alpha=0.7)  # Bar chart
    elif i == 4:
        line = ax.hist(random_noise, bins=20, range=(-3, 3), color='purple', alpha=0.7)  # Histogram
    elif i == 5:
        line = ax.pie([10, 20, 30, 40], startangle=0, autopct='%1.1f%%', colors=['gold', 'lightblue', 'lightgreen', 'salmon'])  # Pie chart
    elif i == 6:
        line, = ax.plot(polar_theta, np.sin(2 * polar_theta), color='darkred', lw=2)  # Polar plot
    elif i == 7:
        line, = ax.plot(x, np.exp(-x), color='darkgreen', lw=2)  # Exponential decay
    elif i == 8:
        line, = ax.plot(x, np.log(x + 1), color='brown', lw=2)  # Logarithmic curve
    elif i == 9:
        line, = ax.plot(np.sin(x), np.cos(x), color='purple', lw=2)  # Spiral
    elif i == 10:
        line, = ax.plot(x, x**2, color='red', lw=2)  # Quadratic
    elif i == 11:
        line, = ax.plot(x, random_noise, color='cyan', lw=2)  # Random noise
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
            ax.plot(scatter_data[0], scatter_data[1], 'go', alpha=0.7)  # Update scatter plot
            ax.set_title(plot_titles[i], fontsize=12, fontweight='bold')
            ax.grid(True)
        elif i == 3:
            bar_data = np.roll(bar_data, shift=1)
            for bar, height in zip(lines[i], bar_data):
                bar.set_height(height)  # Update bar chart
        elif i == 4:
            random_noise = np.random.randn(100)
            ax.clear()
            ax.hist(random_noise, bins=20, range=(-3, 3), color='purple', alpha=0.7)  # Update histogram
            ax.set_title(plot_titles[i], fontsize=12, fontweight='bold')
            ax.grid(True)
        elif i == 5:
            ax.clear()
            ax.pie([10, 20, 30, 40], startangle=frame, autopct='%1.1f%%', colors=['gold', 'lightblue', 'lightgreen', 'salmon'])  # Update pie chart
            ax.set_title(plot_titles[i], fontsize=12, fontweight='bold')
        elif i == 6:
            lines[i].set_ydata(np.sin(2 * polar_theta + frame * 0.1))  # Update polar plot
        elif i == 7:
            lines[i].set_ydata(np.exp(-x + frame * 0.01))  # Update exponential decay
        elif i == 8:
            lines[i].set_ydata(np.log(x + 1 + frame * 0.01))  # Update logarithmic curve
        elif i == 9:
            ax.clear()
            ax.plot(np.sin(x + frame * 0.1), np.cos(x + frame * 0.1), color='purple', lw=2)  # Update spiral
            ax.set_title(plot_titles[i], fontsize=12, fontweight='bold')
            ax.grid(True)
        elif i == 10:
            ax.clear()
            ax.plot(x, (x + frame * 0.01)**2, color='red', lw=2)  # Update quadratic
            ax.set_title(plot_titles[i], fontsize=12, fontweight='bold')
            ax.grid(True)
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
