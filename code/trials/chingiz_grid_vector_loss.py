import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# Define the function to process each file, count cycles, and calculate DTW error
def process_and_plot_file(file_path, file_idx, delta_column_x, delta_column_y, \
                          start_index=0, end_index=None, grid_size=10, min_cycle_length=30, \
                            max_cycle_length=300, epsilon=25.0):
    # Read the CSV file containing the delta values
    delta_data = pd.read_csv(file_path)

    # Extract delta values for x and y
    delta_x = delta_data[delta_column_x][start_index:end_index].values
    delta_y = delta_data[delta_column_y][start_index:end_index].values

    # Initial point (starting point)
    initial_x = 0
    initial_y = 0

    # Create the delta-based path by accumulating delta values
    delta_path_x = [initial_x]
    delta_path_y = [initial_y]

    # Accumulate delta values to create the path
    for dx, dy in zip(delta_x, delta_y):
        next_x = delta_path_x[-1] + dx
        next_y = delta_path_y[-1] + dy
        delta_path_x.append(next_x)
        delta_path_y.append(next_y)

    # Define grid boundaries
    x_min, x_max = min(delta_path_x), max(delta_path_x)
    y_min, y_max = min(delta_path_y), max(delta_path_y)

    dis = grid_size / 2
    # Calculate grid lines
    x_grid_lines = np.arange(x_min - grid_size + dis, x_max + grid_size + dis, grid_size)
    y_grid_lines = np.arange(y_min - grid_size + dis, y_max + grid_size + dis, grid_size)

    # Initialize lists to store the quantized path
    quantized_x = []
    quantized_y = []

    # Quantize the delta-based path using the consistent grid size
    for x, y in zip(delta_path_x, delta_path_y):
        x_cell = np.searchsorted(x_grid_lines, x) - 1
        y_cell = np.searchsorted(y_grid_lines, y) - 1

        if 0 <= x_cell < len(x_grid_lines) and 0 <= y_cell < len(y_grid_lines):
            x_center = x_grid_lines[x_cell] + grid_size / 2
            y_center = y_grid_lines[y_cell] + grid_size / 2

            if not quantized_x or (x_center != quantized_x[-1] or y_center != quantized_y[-1]):
                quantized_x.append(x_center)
                quantized_y.append(y_center)

    # Combine x and y coordinates into tuples for each point in both paths
    original_path = list(zip(delta_path_x, delta_path_y))
    quantized_path = list(zip(quantized_x, quantized_y))
    print(f"delta path: {delta_path_x}, {delta_path_y}\n")
    print(f"quan path: {quantized_x}, {quantized_y}") 
    # Cycle Counting Function
    def count_cycles_with_threshold(quantized_path, grid_size, min_cycle_length=30, max_cycle_length=300, epsilon=25.0):
        visited = {}  # Dictionary to store when each grid point was first visited
        cycles = []  # List to store the indices of the cycles
        cycle_count = 0  # Total cycle count
        i = 0

        def euclidean_distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        while i < len(quantized_path):
            point = quantized_path[i]
            detected_cycle = False

            # Instead of an exact match, check if the point is close to a previously visited point
            for visited_point, visited_index in visited.items():
                if euclidean_distance(point, visited_point) <= epsilon:
                    cycle_start = visited_index  # The first time this close point was visited
                    cycle_end = i  # The current index of the revisited point

                    cycle_length = cycle_end - cycle_start

                    # Check if cycle length falls within the specified range
                    if min_cycle_length <= cycle_length <= max_cycle_length:
                        cycles.append((cycle_start, cycle_end))
                        cycle_count += 1
                        detected_cycle = True

                        # Clear all points except the last one
                        visited.clear()
                        visited[point] = cycle_end  # Keep only the last point

                    break  # Exit the loop once a cycle is detected

            if not detected_cycle:
                # Store the index of the first visit to this point if no cycle was detected
                visited[point] = i

            i += 1

        return cycles, cycle_count

    # Function to find the closest point on a segment
    def find_closest_point_on_segment(px, py, ax, ay, bx, by):
        ABx, ABy = bx - ax, by - ay
        APx, APy = px - ax, py - ay
        dot_product = ABx * APx + ABy * APy
        AB_len_sq = ABx ** 2 + ABy ** 2

        if AB_len_sq == 0:
            return (ax, ay)

        t = max(0, min(1, dot_product / AB_len_sq))
        closest_x, closest_y = ax + t * ABx, ay + t * ABy

        return (closest_x, closest_y)

    # Function to compare raw points to closest segments and calculate total difference
    def compare_raw_points_to_segments(raw_path, quantized_path):
        total_difference = 0  # Initialize total difference
        comparison_pairs = []
        for (px, py) in raw_path:
            min_distance = float('inf')
            closest_point_on_segment = None

            # Compare point to all segments in the quantized path
            for (x1, y1), (x2, y2) in zip(quantized_path[:-1], quantized_path[1:]):
                closest_point = find_closest_point_on_segment(px, py, x1, y1, x2, y2)
                distance = np.sqrt((px - closest_point[0]) ** 2 + (py - closest_point[1]) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    closest_point_on_segment = closest_point

            total_difference += min_distance
            comparison_pairs.append(((px, py), closest_point_on_segment, min_distance))
        
        return comparison_pairs, total_difference

    # Perform raw point to segment comparison and calculate total difference
    comparison_results, total_difference = compare_raw_points_to_segments(original_path, quantized_path)

    # Function to calculate the total length of a path
    def calculate_path_length(path):
        total_length = 0
        for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            total_length += distance
        return total_length

    # Calculate the total length of the original path
    original_path_length = calculate_path_length(original_path)

    # Calculate the percentage error (DTW error percentage)
    if original_path_length > 0:
        percentage_error = (total_difference / original_path_length) * 100
    else:
        percentage_error = 0

    # Perform cycle counting
    detected_cycles, cycle_count = count_cycles_with_threshold(quantized_path, grid_size, \
                                                               min_cycle_length=min_cycle_length, \
                                                                max_cycle_length=max_cycle_length, epsilon=epsilon)

    # Print DTW error percentage and cycle count for each file
    print(f"File {file_idx}: DTW Error Percentage: {percentage_error:.2f}%")
    print(f"Total difference: {total_difference:.2f}")
    print(f"Length of raw path: {original_path_length:.2f}")
    print(f"File {file_idx}: Detected {cycle_count} cycles.")
    

    # Plotting raw points and quantized path with closest segment connections
    plt.figure(figsize=(5, 3))
    plt.plot(delta_path_x, delta_path_y, 'ko-', label='Original Delta-Based Path', alpha=0.6)
    plt.plot(quantized_x, quantized_y, 'g-', linewidth=2, label=f'Quantized Path (Grid Size: {grid_size})')

    # Highlight each cycle with a different color
    def generate_unique_color():
        return (random.random(), random.random(), random.random())

    for idx, (start, end) in enumerate(detected_cycles):
        cycle_x = quantized_x[start:end+1]
        cycle_y = quantized_y[start:end+1]
        color = generate_unique_color()  # Generate a unique color for each cycle
        plt.plot(cycle_x, cycle_y, '-', linewidth=2, color=color)
        print(f"Cycle {idx+1}: from index {start} to {end}")

    # Plot alignment lines based on closest segment
    for (raw_point, closest_point, distance) in comparison_results:
        plt.plot([raw_point[0], closest_point[0]], [raw_point[1], closest_point[1]], 'r-', alpha=0.3)

    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title(f'File {file_idx}: Dist from Every Raw Points to Quantized Segments (Grid size = {grid_size})')
    plt.grid(which='both', linestyle='--', linewidth=0.7)
    plt.xticks(x_grid_lines)
    plt.yticks(y_grid_lines)
    plt.gca().invert_yaxis()
    plt.legend()

    plt.tight_layout()
    plt.show()

# List of file paths (update these to your actual file paths)
file_paths = [
    '/home/chingiz/yolov8-project/difference_csv/delta0.csv'
    # '/home/chingiz/yolov8-project/difference_csv/delta1.csv',
    # '/home/chingiz/yolov8-project/difference_csv/delta2.csv',
    # '/home/chingiz/yolov8-project/difference_csv/delta3.csv',
    # '/home/chingiz/yolov8-project/difference_csv/delta4.csv',
    # '/home/chingiz/yolov8-project/difference_csv/delta5.csv',
    # '/home/chingiz/yolov8-project/difference_csv/delta6.csv',
    # '/home/chingiz/yolov8-project/difference_csv/delta7.csv',
    # '/home/chingiz/yolov8-project/difference_csv/delta8.csv',
    # '/home/chingiz/yolov8-project/difference_csv/delta9.csv'
]

# Loop over all the files and process each
for idx, file_path in enumerate(file_paths):
    process_and_plot_file(file_path, file_idx=idx, delta_column_x=f'dX{idx}',\
                          delta_column_y=f'dY{idx}', start_index=0, end_index=50, grid_size=10)
