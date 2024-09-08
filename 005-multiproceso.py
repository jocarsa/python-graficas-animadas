import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import os
import time
from datetime import datetime, timedelta
import random
import multiprocessing as mp
from multiprocessing import Manager, Pool

# Define video parameters
fps = 60
duration = 60  # 1 minute
total_frames = fps * duration
frame_size = (1920, 1080)

# Create the output directory
output_dir = f'render'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'{int(time.time())}_multiprocessing_control_panel_animation.mp4')

# Generate a random base color
base_color = np.array([random.randint(0, 255) for _ in range(3)]) / 255.0

# Create a figure and 4x3 grid of subplots with black background
fig, axes = plt.subplots(3, 4, figsize=(19.2, 10.8), facecolor='black')
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
    ax.set_title(plot_titles[i], fontsize=12, fontweight='bold', color='white')
    ax.grid(True, color=base_color * 0.2)
    ax.set_facecolor('black')

# Create initial plots with the base color
lines = []
for i, ax in enumerate(axes.flat):
    color = base_color * (0.5 + 0.5 * np.random.rand())  # Different shades of the base color
    if i == 0:
        line, = ax.plot(x, np.sin(x), color=color, lw=2)  # Sine wave
    elif i == 1:
        line, = ax.plot(x, np.cos(x), color=color, lw=2)  # Cosine wave
    elif i == 2:
        line, = ax.plot(scatter_data[0], scatter_data[1], 'o', color=color, alpha=0.7)  # Scatter plot
    elif i == 3:
        line = ax.bar(np.arange(10), bar_data, color=color, alpha=0.7)  # Bar chart
    elif i == 4:
        line = ax.hist(random_noise, bins=20, range=(-3, 3), color=color, alpha=0.7)  # Histogram
    elif i == 5:
        line = ax.pie([10, 20, 30, 40], startangle=0, autopct='%1.1f%%', colors=[color, color*0.8, color*0.6, color*0.4])  # Pie chart
    elif i == 6:
        line, = ax.plot(polar_theta, np.sin(2 * polar_theta), color=color, lw=2)  # Polar plot
    elif i == 7:
        line, = ax.plot(x, np.exp(-x), color=color, lw=2)  # Exponential decay
    elif i == 8:
        line, = ax.plot(x, np.log(x + 1), color=color, lw=2)  # Logarithmic curve
    elif i == 9:
        line, = ax.plot(np.sin(x), np.cos(x), color=color, lw=2)  # Spiral
    elif i == 10:
        line, = ax.plot(x, x**2, color=color, lw=2)  # Quadratic
    elif i == 11:
        line, = ax.plot(x, random_noise, color=color, lw=2)  # Random noise
    lines.append(line)

# Update function for animation
def update_chart(args):
    i, frame, shared_dict = args

    color = base_color * (0.5 + 0.5 * np.random.rand())  # Different shades of the base color
    if i == 0:
        shared_dict[i] = np.sin(x + frame * 0.1)  # Sine wave
    elif i == 1:
        shared_dict[i] = np.cos(x + frame * 0.1)  # Cosine wave
    elif i == 2:
        scatter_data[0] += np.random.randn(100) * 0.01
        scatter_data[1] += np.random.randn(100) * 0.01
        shared_dict[i] = (scatter_data[0], scatter_data[1], color)  # Scatter plot
    elif i == 3:
        bar_data = np.roll(bar_data, shift=1)
        shared_dict[i] = bar_data  # Bar chart
    elif i == 4:
        random_noise = np.random.randn(100)
        shared_dict[i] = random_noise  # Histogram
    elif i == 5:
        shared_dict[i] = frame  # Pie chart
    elif i == 6:
        shared_dict[i] = np.sin(2 * polar_theta + frame * 0.1)  # Polar plot
    elif i == 7:
        shared_dict[i] = np.exp(-x + frame * 0.01)  # Exponential decay
    elif i == 8:
        shared_dict[i] = np.log(x + 1 + frame * 0.01)  # Logarithmic curve
    elif i == 9:
        shared_dict[i] = (np.sin(x + frame * 0.1), np.cos(x + frame * 0.1), color)  # Spiral
    elif i == 10:
        shared_dict[i] = (x, (x + frame * 0.01)**2, color)  # Quadratic
    elif i == 11:
        random_noise = np.random.randn(100)
        shared_dict[i] = random_noise  # Random noise

def update(frame):
    manager = Manager()
    shared_dict = manager.dict()

    # Prepare the arguments for each subplot
    args = [(i, frame, shared_dict) for i in range(len(axes.flat))]

    with Pool(mp.cpu_count()) as pool:
        pool.map(update_chart, args)

    # Apply the updates to the actual plots
    for i, ax in enumerate(axes.flat):
        if i in [0, 1, 6, 7, 8]:
            lines[i].set_ydata(shared_dict[i])  # Update sine, cosine, polar, exp, log
        elif i == 2:
            ax.clear()
            ax.plot(shared_dict[i][0], shared_dict[i][1], 'o', color=shared_dict[i][2], alpha=0.7)  # Scatter plot
            ax.set_title(plot_titles[i], fontsize=12, fontweight='bold', color='white')
            ax.grid(True, color=base_color * 0.2)
            ax.set_facecolor('black')
        elif i == 3:
            bar_data = shared_dict[i]
            for bar, height in zip(lines[i], bar_data):
                bar.set_height(height)  # Update bar chart
        elif i == 4:
            ax.clear()
            ax.hist(shared_dict[i], bins=20, range=(-3, 3), color=base_color, alpha=0.7)  # Histogram
            ax.set_title(plot_titles[i], fontsize=12, fontweight='bold', color='white')
            ax.grid(True, color=base_color * 0.2)
            ax.set_facecolor('black')
        elif i == 5:
            ax.clear()
            ax.pie([10, 20, 30, 40], startangle=shared_dict[i], autopct='%1.1f%%', colors=[base_color, base_color*0.8, base_color*0.6, base_color*0.4])  # Pie chart
            ax.set_title(plot_titles[i], fontsize=12, fontweight='bold', color='white')
        elif i == 9:
            ax.clear()
            ax.plot(shared_dict[i][0], shared_dict[i][1], color=shared_dict[i][2], lw=2)  # Spiral
            ax.set_title(plot_titles[i], fontsize=12, fontweight='bold', color='white')
            ax.grid(True, color=base_color * 0.2)
            ax.set_facecolor('black')
        elif i == 10:
            ax.clear()
            ax.plot(shared_dict[i][0], shared_dict[i][1], color=shared_dict[i][2], lw=2)  # Quadratic
            ax.set_title(plot_titles[i], fontsize=12, fontweight='bold', color='white')
            ax.grid(True, color=base_color * 0.2)
            ax.set_facecolor('black')
        elif i == 11:
            lines[i].set_ydata(shared_dict[i])  # Update random noise

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
    
    # Invert the colors
    img = cv2.bitwise_not(img)
    
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
