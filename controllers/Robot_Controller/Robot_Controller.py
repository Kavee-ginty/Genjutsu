from controller import Robot, Lidar
import math

# Initialize Robot and Lidar
robot = Robot()
timestep = int(robot.getBasicTimeStep())

lidar = robot.getDevice('lidar') # Ensure the name matches your Webots world file
lidar.enable(timestep)
lidar.enablePointCloud()

# Configuration constants
FOV = 6.283185  # 2 * PI
SLICE_WIDTH_DEG = 20  # Total width of each active sector
NUM_SECTORS = 8
FRONT_OFFSET = 0 # Adjust if your lidar's 0 index isn't front-facing

def get_sector_data():
    # Get raw range data (list of floats)
    ranges = lidar.getRangeImage()
    n_points = len(ranges)
    
    if n_points == 0:
        return {}

    # Map your labels to their center angles (0 = Front)
    sector_centers = {
        'a': 0,   'b': 45,  'c': 90,  'd': 135,
        'e': 180, 'f': 225, 'g': 270, 'h': 315
    }
    
    results = {}
    half_slice = SLICE_WIDTH_DEG / 2

    for label, center in sector_centers.items():
        sector_points = []
        
        # Calculate angle boundaries for this sector
        start_angle = (center - half_slice) % 360
        end_angle = (center + half_slice) % 360
        
        for i in range(n_points):
            # Calculate the angle for the current index i
            # Webots Lidar usually fills indices clockwise or counter-clockwise
            # This assumes index 0 is Front (0 deg) and increases clockwise
            angle = (i / n_points) * 360
            
            # Handling the wrap-around for the Front sector (a)
            is_in_sector = False
            if start_angle > end_angle: # Wrap case (e.g., 350 to 10)
                if angle >= start_angle or angle <= end_angle:
                    is_in_sector = True
            else:
                if start_angle <= angle <= end_angle:
                    is_in_sector = True
            
            if is_in_sector:
                # Ignore 'inf' or 0 values which represent out of range
                if 0 < ranges[i] < float('inf'):
                    sector_points.append(ranges[i])
        
        # Calculate the average distance for the sector
        if sector_points:
            results[label] = sum(sector_points) / len(sector_points)
        else:
            results[label] = None # No valid points in this slice
            
    return results

# Main Loop
while robot.step(timestep) != -1:
    sector_distances = get_sector_data()
    
    # Example: Accessing specific walls
    front_dist = sector_distances.get('a')
    left_dist = sector_distances.get('g')
    
    print("-" * 30)
    for key, val in sector_distances.items():
        dist_str = f"{val:.3f}m" if val else "No Data"
        print(f"Sector {key}: {dist_str}")