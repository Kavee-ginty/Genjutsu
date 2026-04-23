import cv2
import numpy as np
import math
from controller import Robot

# ==============================================================================
# 1. INITIALIZATION & CONSTANTS
# ==============================================================================
robot = Robot()
timestep = int(robot.getBasicTimeStep())

tile_length = 0.25
wheel_radius = 0.0205  # Adjust if your robot's wheel radius is different (in meters)

# Global variables for tracking
current_pos = [0, 0]
prev_gps_vals = [0, 0, 0]

print("E-puck: Controller initialized. Setting up devices...")

# ==============================================================================
# 2. DEVICE SETUP (Motors & Sensors)
# ==============================================================================
# Emitter (For sending messages to supervisor)
emitter = robot.getDevice('emitter_2')

# Camera
cam = robot.getDevice("camera")
cam.enable(timestep)

# GPS
gps = robot.getDevice("gps")
gps.enable(timestep)

# Compass
compass = robot.getDevice('compass')
if compass:
    compass.enable(timestep)
else:
    print("ERROR: Compass device not found in Scene Tree!")

# lidar
lidar = robot.getDevice('lidar')
lidar.enable(timestep)
lidar.enablePointCloud()  # Optional, for 3D points

# Motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# ==============================================================================
# 3. COMPUTER VISION SETUP (ArUco)
# ==============================================================================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


# ==============================================================================
# 4. SENSOR & COMMUNICATION FUNCTIONS
# ==============================================================================
def send_message(message_string):
    """Sends a string message to the supervisor."""
    binary_data = message_string.encode('utf-8')
    result = emitter.send(binary_data)
    if result == 1:
        print(f"E-puck: '{message_string}' sent successfully!")
        return True
    else:
        return False

def scan_aruco_tag(camera, detector):
    """Captures camera image, detects ArUco tags, and returns processed data."""
    raw_image = camera.getImage()
    if not raw_image:
        return None, None, None, None

    # Convert to NumPy array
    frame = np.frombuffer(raw_image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    
    # Detect markers
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is not None:
        tag_id = int(ids[0][0])
        binary_str = format(tag_id, '08b')
        x = (tag_id >> 4) & 0x0F
        y = tag_id & 0x0F
        return x, y, tag_id, binary_str
    
    return None, None, None, None

def getXYfromgps(gps):
    """Calculates the maze tile coordinates (0-11 grid) from GPS."""
    gps_vals = gps.getValues()
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
    
    return [new_x, new_y]

# ==============================================================================
# 5. MOVEMENT FUNCTIONS
# ==============================================================================
def moveForwardOneTile(gps):
    """Drives the robot forward exactly one maze tile."""
    tolerance = 0.01  # GPS noise margin
    start_vals = gps.getValues()
    start_x, start_y = start_vals[0], start_vals[1]

    speed = 3.0
    left_motor.setVelocity(speed)
    right_motor.setVelocity(speed)

    while True:
        robot.step(timestep)
        gps_vals = gps.getValues()
        current_x, current_y = gps_vals[0], gps_vals[1]

        # Euclidean distance moved
        dx = current_x - start_x
        dy = current_y - start_y
        distance = (dx**2 + dy**2) ** 0.5

        if distance >= tile_length - tolerance:
            break

    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

def moveBackwardOneTile(gps):
    """Drives the robot backward exactly one maze tile."""
    tolerance = 0.01  # GPS noise margin
    start_vals = gps.getValues()
    start_x, start_y = start_vals[0], start_vals[1]

    speed = -3.0
    left_motor.setVelocity(speed)
    right_motor.setVelocity(speed)

    while True:
        robot.step(timestep)
        gps_vals = gps.getValues()
        current_x, current_y = gps_vals[0], gps_vals[1]

        # Euclidean distance moved
        dx = current_x - start_x
        dy = current_y - start_y
        distance = (dx**2 + dy**2) ** 0.5

        if distance >= tile_length - tolerance:
            break

    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    
def turnLeft90(compass):
    """Turns the robot counterclockwise by exactly 90 degrees."""
    initial = compass.getValues()
    initial_heading = math.atan2(initial[0], initial[1])
    target_heading = initial_heading + math.pi / 2
    
    def normalize(angle):
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle
        
    target_heading = normalize(target_heading)

    speed = 2.0
    left_motor.setVelocity(-speed)
    right_motor.setVelocity(speed)

    while True:
        robot.step(timestep)
        current = compass.getValues()
        current_heading = math.atan2(current[0], current[1])
        diff = normalize(current_heading - target_heading)
        
        if abs(diff) < 0.03:  # ~1.7 degrees tolerance
            break
            
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    
def turnRight90(compass):
    """Turns the robot clockwise by exactly 90 degrees."""
    initial = compass.getValues()
    initial_heading = math.atan2(initial[0], initial[1])
    target_heading = initial_heading - math.pi / 2
    
    def normalize(angle):
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle
        
    target_heading = normalize(target_heading)

    speed = 2.0
    left_motor.setVelocity(speed)
    right_motor.setVelocity(-speed)

    while True:
        robot.step(timestep)
        current = compass.getValues()
        current_heading = math.atan2(current[0], current[1])
        diff = normalize(current_heading - target_heading)
        
        if abs(diff) < 0.03:  # ~1.7 degrees tolerance
            break
            
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)


# ==============================================================================
# 6. MAIN EXECUTION LOOP
# ==============================================================================

while robot.step(timestep) != -1:
    
    # 1. Initial Scan & Coordinate Check
    tag_x, tag_y, current_id, tag_bin = scan_aruco_tag(cam, detector)
    maze_coord = getXYfromgps(gps)
    
    if current_id is not None:
        print(f"--- Tag Detected ---")
        print(f"ID: {current_id} | Binary: {tag_bin}")
        print(f"Coordinates -> X: {tag_x}, Y: {tag_y}")
    else:
        pass

    # 2. Hardcoded Movement Sequence
    # scan_aruco_tag(cam, detector)
    
    # moveForwardOneTile(gps)
    # moveForwardOneTile(gps)
    # moveForwardOneTile(gps)
    # moveForwardOneTile(gps)
    
    # turnLeft90(compass)
    # moveForwardOneTile(gps)
    # turnRight90(compass)
    
    # scan_aruco_tag(cam, detector)
    # moveForwardOneTile(gps)
    
    # scan_aruco_tag(cam, detector)
    # moveForwardOneTile(gps)
    
    # scan_aruco_tag(cam, detector)
    # moveForwardOneTile(gps)
    
    # scan_aruco_tag(cam, detector)
    # moveForwardOneTile(gps)
    
    # End program after sequence completes

    # Inside your while loop:
    # Inside your while loop:



    break