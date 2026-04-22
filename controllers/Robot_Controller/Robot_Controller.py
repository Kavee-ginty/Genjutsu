import cv2
import numpy as np
from controller import Robot

def scan_aruco_tag(camera, detector):
    """
    Captures camera image, detects ArUco tags, and returns 
    processed data for the first tag found.
    """
    raw_image = camera.getImage()
    if not raw_image:
        return None, None, None, None

    # Convert to NumPy array
    frame = np.frombuffer(raw_image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    
    # Detect markers
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is not None:
        # Get the first detected ID
        tag_id = int(ids[0][0])
        
        # Format as 8-bit binary string
        binary_str = format(tag_id, '08b')
        
        # Split bits (High 4 bits = x, Low 4 bits = y)
        x = (tag_id >> 4) & 0x0F
        y = tag_id & 0x0F
        
        return x, y, tag_id, binary_str
    
    return None, None, None, None


# Global variable to track current position in discrete maze coordinates
current_pos = [0, 0]

def getXYfromgps(gps):
    """
    Retrieves the current GPS values and calculates the maze tile coordinates.
    Prints the current state of the robot.
    Returns: A list [x, y] representing the current tile index (0-11).
    """
    gps_vals = gps.getValues()
    
    tile_length = 0.25
    start_gps_x = 1.380
    start_gps_y = -1.388
    
    # Calculate offset from start GPS position
    rel_x = start_gps_x - gps_vals[0]
    rel_y = gps_vals[1] - start_gps_y
    
    # Map distance to grid index 0-11
    new_x = int(round(rel_x / tile_length))
    new_y = int(round(rel_y / tile_length))
    
    # Clip results to 0-11 for 12x12 maze
    new_x = max(0, min(new_x, 11))
    new_y = max(0, min(new_y, 11))
    
    # Output the current state to the terminal
    
    
    return [new_x, new_y]

# 4. Main Loop
prev_gps_vals = [0, 0, 0]

# --- Main Setup ---
robot = Robot()
timestep = int(robot.getBasicTimeStep())

cam = robot.getDevice("camera")
cam.enable(timestep)

gps = robot.getDevice("gps")
gps.enable(timestep)

# ArUco Setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# --- Main Loop ---
while robot.step(timestep) != -1:
    # Use the function to get variables
    tag_x, tag_y, current_id, tag_bin = scan_aruco_tag(cam, detector)
    maze_coord = getXYfromgps(gps)
    
    if current_id is not None:
        print(f"--- Tag Detected ---")
        print(f"ID: {current_id} | Binary: {tag_bin}")
        print(f"Coordinates -> X: {tag_x}, Y: {tag_y}")
    else:
        # Optional: Print something or stay quiet when no tag is seen
        pass

    print(f"X={maze_coord[0]},Y={maze_coord[1]};")
    