from controller import Robot, Lidar

# Initialize Robot and Lidar
robot = Robot()
timestep = int(robot.getBasicTimeStep())

lidar = robot.getDevice('lidar')
lidar.enable(timestep)
lidar.enablePointCloud()

# --- PRECISE DISTANCE THRESHOLDS ---
NEAR_WALL_MAX = 0.20  # Inner walls (~0.125m)
FAR_WALL_MIN  = 0.30  # Next tile's far walls (~0.375m)
FAR_WALL_MAX  = 0.45  

def get_avg_dist(sector):
    """Safely calculates average distance, ignoring 'inf' out-of-range values."""
    valid_rays = [r for r in sector if r != float('inf') and r > 0]
    return sum(valid_rays) / len(valid_rays) if valid_rays else float('inf')

def scan_8_walls(ranges, f, N):
    walls = {}
    
    # --- 1. SLICE ONLY THE 4 ORTHOGONAL SECTORS ---
    s_front = ranges[int(170*f) : int(190*f)]
    s_back  = ranges[int(350*f):N] + ranges[0:int(10*f)] # Wrap around
    s_left  = ranges[int(80*f)  : int(100*f)]
    s_right = ranges[int(260*f) : int(280*f)]

    # --- 2. EVALUATE WITH OCCLUSION LOGIC ---
    
    # FRONT
    d_front = get_avg_dist(s_front)
    if d_front < NEAR_WALL_MAX:
        walls['front']       = "WALL"
        walls['front_front'] = "UNKNOWN"
    else:
        walls['front']       = "OPEN"
        walls['front_front'] = "WALL" if (FAR_WALL_MIN < d_front < FAR_WALL_MAX) else "OPEN"

    # BACK
    d_back = get_avg_dist(s_back)
    if d_back < NEAR_WALL_MAX:
        walls['back']      = "WALL"
        walls['back_back'] = "UNKNOWN"
    else:
        walls['back']      = "OPEN"
        walls['back_back'] = "WALL" if (FAR_WALL_MIN < d_back < FAR_WALL_MAX) else "OPEN"

    # LEFT
    d_left = get_avg_dist(s_left)
    if d_left < NEAR_WALL_MAX:
        walls['left']      = "WALL"
        walls['left_left'] = "UNKNOWN"
    else:
        walls['left']      = "OPEN"
        walls['left_left'] = "WALL" if (FAR_WALL_MIN < d_left < FAR_WALL_MAX) else "OPEN"

    # RIGHT
    d_right = get_avg_dist(s_right)
    if d_right < NEAR_WALL_MAX:
        walls['right']       = "WALL"
        walls['right_right'] = "UNKNOWN"
    else:
        walls['right']       = "OPEN"
        walls['right_right'] = "WALL" if (FAR_WALL_MIN < d_right < FAR_WALL_MAX) else "OPEN"

    return walls

# Main Loop
while robot.step(timestep) != -1:
    ranges = lidar.getRangeImage()
    
    if ranges:
        N = len(ranges)
        f = N / 360 
        
        wall_map = scan_8_walls(ranges, f, N)
        
        # --- CONSOLE OUTPUT ---
        print("\n" + "="*35)
        print("     8-WALL ORTHOGONAL SCANNER     ")
        print("="*35)
        
        # Helper to format the output nicely
        def format_status(state):
            if state == "WALL": return "[### WALL ###]"
            if state == "OPEN": return "[   OPEN   ]"
            return "[ ? UNKNOWN]"
        
        print("\n--- Center Tile ---")
        print(f"Front         : {format_status(wall_map['front'])}")
        print(f"Back          : {format_status(wall_map['back'])}")
        print(f"Left          : {format_status(wall_map['left'])}")
        print(f"Right         : {format_status(wall_map['right'])}")
        
        print("\n--- Neighbor Tiles (Look-Ahead) ---")
        print(f"Front-Front   : {format_status(wall_map['front_front'])}")
        print(f"Back-Back     : {format_status(wall_map['back_back'])}")
        print(f"Left-Left     : {format_status(wall_map['left_left'])}")
        print(f"Right-Right   : {format_status(wall_map['right_right'])}")