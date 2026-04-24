# E-Puck Maze Controller — Continuous Scanning Edition
import math
import cv2
import numpy as np
from controller import Robot
import collections

# -----------------------------------------------------------------------------
# 1. INITIALIZATION & CONSTANTS
# -----------------------------------------------------------------------------

robot = Robot()
timestep_ms = int(robot.getBasicTimeStep())
timestep = timestep_ms / 1000.0

tile_length  = 0.25
wheel_radius = 0.0205

emitter     = robot.getDevice('emitter_2')
cam         = robot.getDevice('camera')
compass     = robot.getDevice('compass')
lidar       = robot.getDevice('lidar')
left_motor  = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_ps  = left_motor.getPositionSensor()
right_ps = right_motor.getPositionSensor()
left_ps.enable(timestep_ms)
right_ps.enable(timestep_ms)

compass.enable(timestep_ms)

lidar.enable(timestep_ms)
lidar.enablePointCloud()

# Enable camera permanently
cam.enable(timestep_ms)
processed_tags = set()

robot.step(timestep_ms)
robot.step(timestep_ms)

initial_heading = 0.0
if compass:
    initial_vals    = compass.getValues()
    initial_heading = math.atan2(initial_vals[0], initial_vals[1])

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

NEAR_WALL_MAX = 0.20   
FAR_WALL_MIN  = 0.30   
FAR_WALL_MAX  = 0.45

# -----------------------------------------------------------------------------
# 2. 8-WALL LIDAR FUNCTIONS
# -----------------------------------------------------------------------------

def get_avg_dist(sector):
    valid = [r for r in sector if r != float('inf') and r > 0]
    return sum(valid) / len(valid) if valid else float('inf')

def scan_8_walls(ranges, f, N):
    walls = {}
    s_front = ranges[int(170*f) : int(190*f)]
    s_back  = ranges[int(350*f):N] + ranges[0:int(10*f)]
    s_left  = ranges[int(80*f)  : int(100*f)]
    s_right = ranges[int(260*f) : int(280*f)]

    d_front = get_avg_dist(s_front)
    if d_front < NEAR_WALL_MAX:
        walls['front']       = "WALL"
        walls['front_front'] = "UNKNOWN"
    else:
        walls['front']       = "OPEN"
        walls['front_front'] = "WALL" if FAR_WALL_MIN < d_front < FAR_WALL_MAX else "OPEN"

    d_back = get_avg_dist(s_back)
    if d_back < NEAR_WALL_MAX:
        walls['back']      = "WALL"
        walls['back_back'] = "UNKNOWN"
    else:
        walls['back']      = "OPEN"
        walls['back_back'] = "WALL" if FAR_WALL_MIN < d_back < FAR_WALL_MAX else "OPEN"

    d_left = get_avg_dist(s_left)
    if d_left < NEAR_WALL_MAX:
        walls['left']      = "WALL"
        walls['left_left'] = "UNKNOWN"
    else:
        walls['left']      = "OPEN"
        walls['left_left'] = "WALL" if FAR_WALL_MIN < d_left < FAR_WALL_MAX else "OPEN"

    d_right = get_avg_dist(s_right)
    if d_right < NEAR_WALL_MAX:
        walls['right']       = "WALL"
        walls['right_right'] = "UNKNOWN"
    else:
        walls['right']       = "OPEN"
        walls['right_right'] = "WALL" if FAR_WALL_MIN < d_right < FAR_WALL_MAX else "OPEN"

    return walls

def get_wall_map():
    robot.step(timestep_ms)
    ranges = lidar.getRangeImage()
    if not ranges: return {}
    N = len(ranges)
    f = N / 360
    return scan_8_walls(ranges, f, N)

def scan_and_register_walls():
    wall_map = get_wall_map()
    if not wall_map: return

    heading_to_delta = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
    sensor_heading_offsets = {'front': 0, 'right': 3, 'back': 2, 'left': 1}

    def register_if_wall(sensor_key, tile_delta_steps):
        if wall_map.get(sensor_key) != "WALL": return
        abs_heading = (current_heading + sensor_heading_offsets[sensor_key]) % 4
        dx, dy = heading_to_delta[abs_heading]
        neighbor = (current_x + dx * tile_delta_steps, current_y + dy * tile_delta_steps)
        here     = (current_x + dx * (tile_delta_steps - 1), current_y + dy * (tile_delta_steps - 1))
        known_walls.add(get_wall_id(here, neighbor))

    for direction in ('front', 'back', 'left', 'right'):
        register_if_wall(direction, 1)

    look_ahead_map = {'front_front': 'front', 'back_back': 'back', 'left_left': 'left', 'right_right': 'right'}
    for far_key, near_key in look_ahead_map.items():
        if wall_map.get(near_key) == "OPEN" and wall_map.get(far_key) == "WALL":
            register_if_wall(near_key, 2)

# -----------------------------------------------------------------------------
# 3. HELPER UTILITIES
# -----------------------------------------------------------------------------

def steps_for_distance(distance, angular_speed):
    linear_speed = abs(angular_speed) * wheel_radius
    if linear_speed <= 0.0: return 0
    return max(1, int(round(distance / linear_speed / timestep)))

def send_message(message_string):
    binary_data = message_string.encode('utf-8')
    return emitter.send(binary_data) == 1

def normalize_angle(angle):
    while angle >  math.pi: angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle

def get_wall_id(node_a, node_b):
    return tuple(sorted([node_a, node_b]))

# -----------------------------------------------------------------------------
# 4. ARUCO TAG DETECTION
# -----------------------------------------------------------------------------

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector   = cv2.aruco.ArucoDetector(aruco_dict, parameters)

def check_camera_quick():
    """Checks the camera continuously. Filters out glitch readings and known tags."""
    robot.step(timestep_ms)
    raw = cam.getImage()
    if not raw: return None
    
    img = np.frombuffer(raw, np.uint8).reshape((cam.getHeight(), cam.getWidth(), 4))
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is not None:
        tag_id = int(ids[0][0])
        
        if tag_id in processed_tags:
            return None
            
        x = (tag_id >> 4) & 0x0F
        y = tag_id & 0x0F
        
        # Anti-Hallucination: If the coordinate is outside a 16x16 maze, ignore it.
        if x >= 16 or y >= 16:
            return None

        processed_tags.add(tag_id)
        binary_str = format(tag_id, '08b')
        return (x, y, tag_id, binary_str)
        
    return None

def scan_aruco_tag(scan_distance=0.1):
    """
    Intelligent close-range scan: Prioritizes walls parallel to the X-axis.
    """
    wall_map = get_wall_map()
    offsets = {'front': 0, 'left': 1, 'right': 3, 'back': 2}
    start_heading = current_heading
    
    # 1. Collect all valid walls we can look at
    walls_to_check = []
    for sensor, offset in offsets.items():
        if wall_map.get(sensor) == "WALL":
            target_h = (start_heading + offset) % 4
            walls_to_check.append({
                'sensor': sensor,
                'target_h': target_h,
                'offset': offset
            })
            
    # 2. Sort to prioritize walls parallel to the X-axis
    # walls_to_check.sort(key=lambda x: x['target_h'] % 2)
    
    # 3. Execute the scan in the prioritized order
    for wall in walls_to_check:
        target_h = wall['target_h']
        sensor = wall['sensor']
        
        if target_h != current_heading:
            turn_to_heading(target_h, speed=6.28, reason=f"Scanning {sensor} wall (Heading {target_h}) for AR Tag")
        
        # First try: Instant check
        res = check_camera_quick()
        if res: return res
        
        # Second try: Back up slightly to widen Camera FOV
        move_distance(-scan_distance, speed=6.28)
        res = check_camera_quick()
        move_distance(scan_distance, speed=6.28)
        
        if res: return res

    return None, None, None, None

# -----------------------------------------------------------------------------
# 5. MOVEMENT PRIMITIVES & DRIFT CORRECTION
# -----------------------------------------------------------------------------

def get_dynamic_speeds(base_speed):
    vals = compass.getValues()
    cur_h = math.atan2(vals[0], vals[1])
    rel_angle = normalize_angle(cur_h - initial_heading)
    multiples = [0.0, math.pi/2, math.pi, -math.pi/2, -math.pi]
    target_angle_rel = min(multiples, key=lambda x: abs(normalize_angle(rel_angle - x)))
    angle_error = normalize_angle(rel_angle - target_angle_rel)
    
    Kp_compass = 4.0
    steering_compass = Kp_compass * angle_error
    
    ranges = lidar.getRangeImage()
    steering_lidar = 0.0
    
    if ranges:
        N = len(ranges)
        f = N / 360
        s_left = ranges[int(80*f) : int(100*f)]
        s_right = ranges[int(260*f) : int(280*f)]
        d_left = get_avg_dist(s_left)
        d_right = get_avg_dist(s_right)
        
        wall_threshold = 0.22 
        ideal_dist = 0.125
        Kp_lidar = 6.0
        
        if not math.isinf(d_left) and d_left < wall_threshold:
            steering_lidar = Kp_lidar * (ideal_dist - d_left)
        elif not math.isinf(d_right) and d_right < wall_threshold:
            steering_lidar = Kp_lidar * (d_right - ideal_dist)

    total_steering = steering_compass + steering_lidar
    left_speed = base_speed + total_steering
    right_speed = base_speed - total_steering
    
    max_spd = 6.28
    left_speed = max(-max_spd, min(max_spd, left_speed))
    right_speed = max(-max_spd, min(max_spd, right_speed))
    
    return left_speed, right_speed

def move_distance(distance, speed=2.0):
    rotation      = distance / wheel_radius
    target_left   = left_ps.getValue()  + rotation
    target_right  = right_ps.getValue() + rotation
    vel = abs(speed)
    left_motor.setVelocity(vel)
    right_motor.setVelocity(vel)
    left_motor.setPosition(target_left)
    right_motor.setPosition(target_right)
    while True:
        if robot.step(timestep_ms) == -1: return
        if (abs(left_ps.getValue()  - target_left)  < 0.01 and
                abs(right_ps.getValue() - target_right) < 0.01):
            break
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    robot.step(timestep_ms)
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))

def _snap_heading(tolerance=0.05):
    """Dynamically corrects orientation to the nearest 90-degree axis without spinning."""
    while True:
        if robot.step(timestep_ms) == -1: break
        
        vals = compass.getValues()
        cur_h = math.atan2(vals[0], vals[1])
        rel_angle = normalize_angle(cur_h - initial_heading)
        multiples = [0.0, math.pi/2, math.pi, -math.pi/2, -math.pi]
        closest = min(multiples, key=lambda x: abs(normalize_angle(rel_angle - x)))
        offset = normalize_angle(rel_angle - closest)

        if abs(offset) <= tolerance:
            break

        if offset > 0:
            left_motor.setVelocity(0.8)
            right_motor.setVelocity(-0.8)
        else:
            left_motor.setVelocity(-0.8)
            right_motor.setVelocity(0.8)

    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)


def turn_left_90(correct=True, tolerance=0.05, speed=2.0, reason="Unknown"):
        
    vals           = compass.getValues()
    cur_h          = math.atan2(vals[0], vals[1])
    target_heading = normalize_angle(cur_h + math.pi / 2)
    
    left_motor.setVelocity(-speed)
    right_motor.setVelocity( speed)
    
    while True:
        if robot.step(timestep_ms) == -1: break
        vals  = compass.getValues()
        h     = math.atan2(vals[0], vals[1])
        error = normalize_angle(h - target_heading)
        if abs(error) < tolerance: break
        
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    
    if correct:
        robot.step(timestep_ms)
        _snap_heading(tolerance)
    

def turn_right_90(correct=True, tolerance=0.05, speed=2.0, reason="Unknown"):
    
    vals           = compass.getValues()
    cur_h          = math.atan2(vals[0], vals[1])
    target_heading = normalize_angle(cur_h - math.pi / 2)
    
    left_motor.setVelocity( speed)
    right_motor.setVelocity(-speed)
    
    while True:
        if robot.step(timestep_ms) == -1: break
        vals  = compass.getValues()
        h     = math.atan2(vals[0], vals[1])
        error = normalize_angle(h - target_heading)
        if abs(error) < tolerance: break
        
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    
    if correct:
        robot.step(timestep_ms)
        _snap_heading(tolerance)


def turn_180(correct=True, tolerance=0.05, speed=2.0, reason="Unknown"):
    
    vals           = compass.getValues()
    cur_h          = math.atan2(vals[0], vals[1])
    target_heading = normalize_angle(cur_h + math.pi) 
    
    left_motor.setVelocity(-speed)
    right_motor.setVelocity( speed)
    
    while True:
        if robot.step(timestep_ms) == -1: break
        vals  = compass.getValues()
        h     = math.atan2(vals[0], vals[1])
        error = normalize_angle(h - target_heading)
        if abs(error) < tolerance: break
        
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    
    if correct:
        robot.step(timestep_ms)
        _snap_heading(tolerance)

def turn_to_heading(target_h, speed=2.0, reason="Unknown"):
    global current_heading
    diff = (target_h - current_heading) % 4
    if diff == 1:
        turn_left_90(speed=speed, reason=reason)
    elif diff == 2:
        turn_180(speed=speed, reason=reason) 
    elif diff == 3:
        turn_right_90(speed=speed, reason=reason)
    current_heading = target_h

def move_forward_tiles(num_tiles, speed=3.0):
    steps_per_tile = steps_for_distance(tile_length, speed)
    for _ in range(num_tiles):
        for _ in range(steps_per_tile):
            adj_left, adj_right = get_dynamic_speeds(speed)
            left_motor.setVelocity(adj_left)
            right_motor.setVelocity(adj_right)
            if robot.step(timestep_ms) == -1: return
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        robot.step(timestep_ms)

def adjust_to_wall(target_distance=0.128, tolerance=0.02, wall_threshold=0.3):
    ranges = lidar.getRangeImage()
    if ranges:
        N = len(ranges)
        f = N / 360
        s_front = ranges[int(170*f) : int(190*f)]
        d = get_avg_dist(s_front)
        if math.isinf(d) or d > wall_threshold: return
        if d >= 0.09 and d <= 0.17: return

    max_speed = 1.5
    min_speed = 0.2
    while True:
        if robot.step(timestep_ms) == -1: break
        ranges = lidar.getRangeImage()
        if not ranges: continue
        N = len(ranges)
        f = N / 360
        s_front = ranges[int(170*f) : int(190*f)]
        front   = get_avg_dist(s_front)
        if math.isinf(front): break
        error = front - target_distance
        if abs(error) <= tolerance: break
        speed = max(min_speed, min(max_speed, abs(error) * 5))
        if error > 0:
            left_motor.setVelocity( speed)
            right_motor.setVelocity( speed)
        else:
            left_motor.setVelocity(-speed)
            right_motor.setVelocity(-speed)
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    robot.step(timestep_ms)
    ranges = lidar.getRangeImage()
    if not ranges: return
    N = len(ranges)
    f = N / 360
    s_left = ranges[int(80*f) : int(100*f)]
    s_right = ranges[int(260*f) : int(280*f)]
    d_left = get_avg_dist(s_left)
    d_right = get_avg_dist(s_right)

    side_aligned = False
    if not math.isinf(d_left) and d_left < wall_threshold:
        turn_left_90(correct=False, reason="Facing Left side-wall to correct drift")
        _adjust_to_side(target_distance, tolerance, wall_threshold)
        turn_right_90(correct=False, reason="Returning to path after Left side-wall alignment")
        side_aligned = True
    elif not math.isinf(d_right) and d_right < wall_threshold:
        turn_right_90(correct=False, reason="Facing Right side-wall to correct drift")
        _adjust_to_side(target_distance, tolerance, wall_threshold)
        turn_left_90(correct=False, reason="Returning to path after Right side-wall alignment")
        side_aligned = True

def _adjust_to_side(target_distance=0.128, tolerance=0.02, wall_threshold=0.3):
    max_speed = 1.5
    min_speed = 0.2
    while True:
        if robot.step(timestep_ms) == -1: break
        ranges = lidar.getRangeImage()
        if not ranges: continue
        N = len(ranges)
        f = N / 360
        s_front = ranges[int(170*f) : int(190*f)]
        front   = get_avg_dist(s_front)
        if math.isinf(front): break
        error = front - target_distance
        if abs(error) <= tolerance: break
        speed = max(min_speed, min(max_speed, abs(error) * 5))
        if error > 0:
            left_motor.setVelocity( speed)
            right_motor.setVelocity( speed)
        else:
            left_motor.setVelocity(-speed)
            right_motor.setVelocity(-speed)
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

def check_wall_ahead(threshold=0.18):
    ranges = lidar.getRangeImage()
    if not ranges: return False
    N = len(ranges)
    f = N / 360
    wall_map = scan_8_walls(ranges, f, N)
    s_front = ranges[int(170*f) : int(190*f)]
    d_front = get_avg_dist(s_front)
    return wall_map['front'] == "WALL" or (not math.isinf(d_front) and d_front < threshold)

# -----------------------------------------------------------------------------
# 7. NAVIGATION
# -----------------------------------------------------------------------------

def get_shortest_path_bfs(start, target, grid_size=16, restrict_to_visited=False):
    from collections import deque
    if start == target: return [start]
    queue   = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nxt = (node[0] + dx, node[1] + dy)
            if not (0 <= nxt[0] < grid_size and 0 <= nxt[1] < grid_size): continue
            
            # --- SAFE BACKTRACKING FILTER ---
            if restrict_to_visited and nxt not in historically_visited: continue
            
            if get_wall_id(node, nxt) in known_walls: continue
            if nxt in visited: continue
            new_path = path + [nxt]
            if nxt == target: return new_path
            visited.add(nxt)
            queue.append((nxt, new_path))
    return None

def get_target_heading(current_node, next_node):
    dx = next_node[0] - current_node[0]
    dy = next_node[1] - current_node[1]
    if dy ==  1: return 0  
    if dx ==  1: return 1   
    if dy == -1: return 2   
    if dx == -1: return 3   
    return 0

def navigate_to_target(target_x, target_y):
    global current_x, current_y, current_heading
    straight_tiles_count = 0
    current_path = [] 

    while (current_x, current_y) != (target_x, target_y):
        scan_and_register_walls()

        if target_x is None:
            return True, None

        # --- CONTINUOUS CAMERA SCANNING ---
        # The camera is always on, and check_camera_quick() filters out known tags.
        # If it returns data here, it's a completely NEW tag spotted in the distance!
        res = check_camera_quick()
        if res:
            print(f">>> Early Tag detected mid-navigation! (X, Y): ({res[0]}, {res[1]})")
            return True, res 

        # --- SMART PATH CACHING & SAFE BACKTRACKING ---
        recalculate = True
        
        if current_path and len(current_path) > 1:
            if (current_x, current_y) in current_path:
                idx = current_path.index((current_x, current_y))
                current_path = current_path[idx:] 
                
                if len(current_path) > 1:
                    blocked = False
                    for i in range(len(current_path) - 1):
                        if get_wall_id(current_path[i], current_path[i+1]) in known_walls:
                            blocked = True
                            break
                    if not blocked:
                        recalculate = False

        if recalculate:
            new_path = None
            
            # 1. Try Safe Backtracking ONLY if the exact target is in our history
            if (target_x, target_y) in historically_visited:
                new_path = get_shortest_path_bfs((current_x, current_y), (target_x, target_y), restrict_to_visited=True)
            
            # 2. If it's an unknown target, OR safe backtracking couldn't find a contiguous path, use Normal BFS
            if not new_path or len(new_path) < 2:
                new_path = get_shortest_path_bfs((current_x, current_y), (target_x, target_y))

            current_path = new_path

            # 3. If still no path, abort
            if not current_path or len(current_path) < 2:
                return False, None

        next_node = current_path[1]
        target_h  = get_target_heading((current_x, current_y), next_node)

        if target_h != current_heading:
            turn_to_heading(target_h, reason=f"BFS Path Correction (Heading to {target_h})")
            straight_tiles_count = 0

        if check_wall_ahead():
            print(f"[BLOCKED] Wall detected ahead! Recalculating route...")
            adjust_to_wall()
            known_walls.add(get_wall_id((current_x, current_y), next_node))
            straight_tiles_count = 0
            current_path = [] 
            continue

        
        move_forward_tiles(1)
        current_x, current_y = next_node
        
        # Add the new tile to our memory bank!
        historically_visited.add((current_x, current_y)) 
        
        straight_tiles_count += 1

    print(f"Reached coordinate: ({target_x}, {target_y})")
    return True, None

# -----------------------------------------------------------------------------
# 8. FINAL DETECTION & MESSAGING
# -----------------------------------------------------------------------------

def detect_final_wall_color():
    global current_heading
    print("\n[COLOR SCAN] Starting Center Pixel Detection...")
    
    offsets = [0, 1, 2, 3] 
    start_heading = current_heading

    for offset in offsets:
        target_h = (start_heading + offset) % 4
        turn_to_heading(target_h, speed=6.28, reason=f"Scanning heading {target_h}")

        for _ in range(6):
            robot.step(timestep_ms)

        raw = cam.getImage()
        if raw:
            img = np.frombuffer(raw, np.uint8).reshape((cam.getHeight(), cam.getWidth(), 4))
            cy, cx = cam.getHeight() // 2, cam.getWidth() // 2
            
            pixel = img[cy, cx]
            b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])
            
            print(f"[COLOR TEST] Heading {target_h} Center Pixel -> R:{r} G:{g} B:{b}")

            if r < 50 and g < 50 and b < 50:
                return "Black"

            if r > 180 and g > 180 and b > 180:
                if abs(r - g) < 15 and abs(g - b) < 15:
                    continue 

            rgb_pixel = np.uint8([[[r, g, b]]])
            hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)[0][0]
            hue, sat, val = hsv_pixel[0], hsv_pixel[1], hsv_pixel[2]

            if sat > 30: 
                if hue < 15 or hue > 165:
                    return "Red"
                elif 40 < hue < 85:
                    return "Green"
                elif 100 < hue < 145:
                    return "Blue"

    return "Unknown"

# -----------------------------------------------------------------------------
# 9. GLOBAL STATE & MAIN
# -----------------------------------------------------------------------------

current_x       = 0
current_y       = 0
current_heading = 0   
known_walls     = set()
historically_visited = {(0, 0)} # Memory of physically visited tiles


def main():
    global current_x, current_y

    robot.step(timestep_ms)

    # --- 1. INITIAL SCAN: Check for a tag straight ahead before moving ---
    for _ in range(5):
        robot.step(timestep_ms)
        
    next_tag_data = check_camera_quick()

    if next_tag_data:
        x, y, tag_id, binary_str = next_tag_data
        print(f">>> Initial Scan: Valid Tag Found straight ahead! Decoded (X, Y): ({x}, {y})")

    # --- 2. ENTER THE MAZE ---
    while not check_wall_ahead():
        move_forward_tiles(1)
        current_y += 1
        
        historically_visited.add((current_x, current_y)) 
        
        scan_and_register_walls()

    adjust_to_wall()

    # --- 3. MAIN NAVIGATION LOOP ---
    while robot.step(timestep_ms) != -1:
        
        if next_tag_data is None:
            x, y, tag_id, binary_str = scan_aruco_tag()
        else:
            x, y, tag_id, binary_str = next_tag_data
            next_tag_data = None 

        if tag_id is not None:
            print(f"Valid Tag Found! Decoded (X, Y): ({x}, {y})")
            
            success, early_tag = navigate_to_target(x, y)
            
            if early_tag:
                next_tag_data = early_tag
                
            if not success:
                break
        else:
            break

    color = detect_final_wall_color()
    print(f"Final Wall Color: {color}")
    send_message(color)
    print("Time taken:", robot.getTime()/60)

if __name__ == '__main__':
    main()