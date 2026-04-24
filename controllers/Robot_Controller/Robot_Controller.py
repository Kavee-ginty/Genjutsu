from controller import Robot, Lidar

# Initialize Robot and Lidar
robot = Robot()
timestep = int(robot.getBasicTimeStep())

lidar = robot.getDevice('lidar')
lidar.enable(timestep)
lidar.enablePointCloud()

# --- PRECISE DISTANCE THRESHOLDS ---
NEAR_WALL_MAX = 0.20  
FAR_ORTHO_MIN = 0.30
FAR_ORTHO_MAX = 0.45
FAR_DIAG_MIN = 0.22
FAR_DIAG_MAX = 0.35

def get_avg_dist(sector):
    """Safely calculates average distance, ignoring 'inf' out-of-range values."""
    valid_rays = [r for r in sector if r != float('inf') and r > 0]
    return sum(valid_rays) / len(valid_rays) if valid_rays else float('inf')

def scan_16_walls(ranges, f, N):
    walls = {}
    
    # --- 1. SLICE ALL SECTORS ---
    # Orthogonal
    s_front = ranges[int(170*f) : int(190*f)]
    s_back  = ranges[int(350*f):N] + ranges[0:int(10*f)]
    s_left  = ranges[int(80*f) : int(100*f)]
    s_right = ranges[int(260*f) : int(280*f)]
    
    # Diagonals (Looking into neighbors)
    s_f_nl = ranges[int(150*f) : int(160*f)] # Front Neighbor's Left Wall
    s_f_nr = ranges[int(200*f) : int(210*f)] # Front Neighbor's Right Wall
    
    s_b_nl = ranges[int(20*f) : int(30*f)]   # Back Neighbor's Left Wall
    s_b_nr = ranges[int(330*f) : int(340*f)] # Back Neighbor's Right Wall
    
    s_l_nf = ranges[int(110*f) : int(120*f)] # Left Neighbor's Front Wall
    s_l_nb = ranges[int(60*f) : int(70*f)]   # Left Neighbor's Back Wall
    
    s_r_nf = ranges[int(240*f) : int(250*f)] # Right Neighbor's Front Wall
    s_r_nb = ranges[int(290*f) : int(300*f)] # Right Neighbor's Back Wall

    # --- 2. EVALUATE WITH OCCLUSION LOGIC ---
    
    # FRONT DIRECTION
    if get_avg_dist(s_front) < NEAR_WALL_MAX:
        walls['Center_Front_Wall']         = "WALL"
        walls['Front_Neighbor_Front_Wall'] = "UNKNOWN"
        walls['Front_Neighbor_Left_Wall']  = "UNKNOWN"
        walls['Front_Neighbor_Right_Wall'] = "UNKNOWN"
    else:
        walls['Center_Front_Wall']         = "OPEN"
        walls['Front_Neighbor_Front_Wall'] = "WALL" if (FAR_ORTHO_MIN < get_avg_dist(s_front) < FAR_ORTHO_MAX) else "OPEN"
        walls['Front_Neighbor_Left_Wall']  = "WALL" if (FAR_DIAG_MIN < get_avg_dist(s_f_nl) < FAR_DIAG_MAX) else "OPEN"
        walls['Front_Neighbor_Right_Wall'] = "WALL" if (FAR_DIAG_MIN < get_avg_dist(s_f_nr) < FAR_DIAG_MAX) else "OPEN"

    # BACK DIRECTION
    if get_avg_dist(s_back) < NEAR_WALL_MAX:
        walls['Center_Back_Wall']         = "WALL"
        walls['Back_Neighbor_Back_Wall']  = "UNKNOWN"
        walls['Back_Neighbor_Left_Wall']  = "UNKNOWN"
        walls['Back_Neighbor_Right_Wall'] = "UNKNOWN"
    else:
        walls['Center_Back_Wall']         = "OPEN"
        walls['Back_Neighbor_Back_Wall']  = "WALL" if (FAR_ORTHO_MIN < get_avg_dist(s_back) < FAR_ORTHO_MAX) else "OPEN"
        walls['Back_Neighbor_Left_Wall']  = "WALL" if (FAR_DIAG_MIN < get_avg_dist(s_b_nl) < FAR_DIAG_MAX) else "OPEN"
        walls['Back_Neighbor_Right_Wall'] = "WALL" if (FAR_DIAG_MIN < get_avg_dist(s_b_nr) < FAR_DIAG_MAX) else "OPEN"

    # LEFT DIRECTION
    if get_avg_dist(s_left) < NEAR_WALL_MAX:
        walls['Center_Left_Wall']         = "WALL"
        walls['Left_Neighbor_Left_Wall']  = "UNKNOWN"
        walls['Left_Neighbor_Front_Wall'] = "UNKNOWN"
        walls['Left_Neighbor_Back_Wall']  = "UNKNOWN"
    else:
        walls['Center_Left_Wall']         = "OPEN"
        walls['Left_Neighbor_Left_Wall']  = "WALL" if (FAR_ORTHO_MIN < get_avg_dist(s_left) < FAR_ORTHO_MAX) else "OPEN"
        walls['Left_Neighbor_Front_Wall'] = "WALL" if (FAR_DIAG_MIN < get_avg_dist(s_l_nf) < FAR_DIAG_MAX) else "OPEN"
        walls['Left_Neighbor_Back_Wall']  = "WALL" if (FAR_DIAG_MIN < get_avg_dist(s_l_nb) < FAR_DIAG_MAX) else "OPEN"

    # RIGHT DIRECTION
    if get_avg_dist(s_right) < NEAR_WALL_MAX:
        walls['Center_Right_Wall']         = "WALL"
        walls['Right_Neighbor_Right_Wall'] = "UNKNOWN"
        walls['Right_Neighbor_Front_Wall'] = "UNKNOWN"
        walls['Right_Neighbor_Back_Wall']  = "UNKNOWN"
    else:
        walls['Center_Right_Wall']         = "OPEN"
        walls['Right_Neighbor_Right_Wall'] = "WALL" if (FAR_ORTHO_MIN < get_avg_dist(s_right) < FAR_ORTHO_MAX) else "OPEN"
        walls['Right_Neighbor_Front_Wall'] = "WALL" if (FAR_DIAG_MIN < get_avg_dist(s_r_nf) < FAR_DIAG_MAX) else "OPEN"
        walls['Right_Neighbor_Back_Wall']  = "WALL" if (FAR_DIAG_MIN < get_avg_dist(s_r_nb) < FAR_DIAG_MAX) else "OPEN"

    return walls

# Main Loop
while robot.step(timestep) != -1:
    ranges = lidar.getRangeImage()
    
    if ranges:
        N = len(ranges)
        f = N / 360 
        
        wall_map = scan_16_walls(ranges, f, N)
        
        print("\n" + "="*45)
        print("         16-WALL OCCLUSION MAPPER         ")
        print("="*45)
        
        categories = {
            "Center Tile": ['Center_Front_Wall', 'Center_Back_Wall', 'Center_Left_Wall', 'Center_Right_Wall'],
            "Front Neighbor": ['Front_Neighbor_Front_Wall', 'Front_Neighbor_Left_Wall', 'Front_Neighbor_Right_Wall'],
            "Left Neighbor": ['Left_Neighbor_Left_Wall', 'Left_Neighbor_Front_Wall', 'Left_Neighbor_Back_Wall'],
            "Right Neighbor": ['Right_Neighbor_Right_Wall', 'Right_Neighbor_Front_Wall', 'Right_Neighbor_Back_Wall'],
            "Back Neighbor": ['Back_Neighbor_Back_Wall', 'Back_Neighbor_Left_Wall', 'Back_Neighbor_Right_Wall']
        }
        
        for category, keys in categories.items():
            print(f"\n--- {category} ---")
            for key in keys:
                state = wall_map[key]
                # Formatting the output to make it easy to read
                if state == "WALL":
                    status = "[### WALL ###]"
                elif state == "OPEN":
                    status = "[   OPEN   ]"
                else:
                    status = "[ ? UNKNOWN]"
                    
                print(f"{key.replace('_', ' '):<28} : {status}")