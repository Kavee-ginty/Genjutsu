# Drift-Proof e-puck controller with PID Compass & Lateral Centering
import math
import cv2
import numpy as np
from controller import Robot
import collections
import heapq
from collections import deque

# -----------------------------------------------------------------------------
# 1. INITIALIZATION & CONSTANTS
# -----------------------------------------------------------------------------

robot = Robot()
timestep_ms = int(robot.getBasicTimeStep())
timestep = timestep_ms / 1000.0

tile_length = 0.25  
wheel_radius = 0.0205  

NEAR_WALL_MAX = 0.18  
FAR_ORTHO_MIN = 0.33  
FAR_ORTHO_MAX = 0.42
FAR_DIAG_MIN  = 0.24  
FAR_DIAG_MAX  = 0.34

emitter = robot.getDevice('emitter_2')
cam = robot.getDevice('camera')
gps = robot.getDevice('gps')
compass = robot.getDevice('compass')
lidar = robot.getDevice('lidar')
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_ps = left_motor.getPositionSensor()
right_ps = right_motor.getPositionSensor()
left_ps.enable(timestep_ms)
right_ps.enable(timestep_ms)

gps.enable(timestep_ms)
compass.enable(timestep_ms)

lidar.enable(timestep_ms)
lidar.enablePointCloud()
cam.disable()    

robot.step(timestep_ms)
initial_heading = 0.0
if compass:
    initial_vals = compass.getValues()
    initial_heading = math.atan2(initial_vals[0], initial_vals[1])

start_gps_x = 1.380
start_gps_y = -1.388

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

def steps_for_distance(distance: float, angular_speed: float) -> int:
    linear_speed = abs(angular_speed) * wheel_radius
    if linear_speed <= 0.0:
        return 0
    travel_time = distance / linear_speed
    return max(1, int(round(travel_time / timestep)))

def send_message(message_string):
    binary_data = message_string.encode('utf-8')
    result = emitter.send(binary_data)
    return result == 1

# -----------------------------------------------------------------------------
# 1.5 DIAGNOSTIC LOGGER (FLIGHT RECORDER)
# -----------------------------------------------------------------------------
action_log = deque(maxlen=10)

def log_action(action_desc: str):
    action_log.append(f"Tile: ({current_x}, {current_y}) | Heading: {current_heading} | Action: {action_desc}")

def dump_crash_log():
    print("\n" + "!" * 55)
    print("      DIAGNOSTIC LOG: LAST 10 ACTIONS      ")
    print("!" * 55)
    if not action_log:
        print("No actions recorded.")
    else:
        for i, entry in enumerate(action_log):
            print(f"{i + 1}. {entry}")
    print("!" * 55 + "\n")


# -----------------------------------------------------------------------------
# 1.8 HARDWARE ABSTRACTION: 360-RAY ALIGNMENT
# -----------------------------------------------------------------------------
def get_360_ranges():
    raw = lidar.getRangeImage()
    if not raw or len(raw) < 360:
        return None
    return raw[::-1]


# -----------------------------------------------------------------------------
# 2. ARUCO TAG DETECTION
# -----------------------------------------------------------------------------

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

def scan_aruco_tag(wall_threshold: float = 0.35, scan_distance: float = 0.15) -> tuple:
    def detect_once() -> tuple:
        cam.enable(timestep_ms)
        for _ in range(5):
            robot.step(timestep_ms)
            
        raw = cam.getImage()
        cam.disable()
        
        if not raw: return None, None, None, None
            
        img = np.frombuffer(raw, np.uint8).reshape((cam.getHeight(), cam.getWidth(), 4))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None:
            tag_id = int(ids[0][0])
            binary_str = format(tag_id, '08b')
            x = (tag_id >> 4) & 0x0F
            y = tag_id & 0x0F
            return x, y, tag_id, binary_str
            
        return None, None, None, None

    robot.step(timestep_ms)
    turns = 0  
    
    for attempt in range(4):
        ranges = get_360_ranges()
        if ranges:
            front_slice = [r for r in ranges[178:183] if not math.isinf(r)]
            front_dist = min(front_slice) if front_slice else float('inf')
            
            if front_dist < wall_threshold:
                result = detect_once()
                if result[2] is None:
                    move_distance(-scan_distance)
                    result = detect_once()
                    move_distance(scan_distance)
                
                if result[2] is not None: 
                    for _ in range(turns): turn_left_90(correct=False)
                    return result

        turn_right_90(correct=False)
        turns += 1
        
    for _ in range(turns): turn_left_90(correct=False)
    
    return None, None, None, None


# -----------------------------------------------------------------------------
# 3. 16-WALL LIDAR MAPPING & OCCLUSION LOGIC
# -----------------------------------------------------------------------------

def get_avg_dist(sector):
    valid_rays = [r for r in sector if not math.isinf(r) and r > 0]
    return sum(valid_rays) / len(valid_rays) if valid_rays else float('inf')

def scan_16_walls(ranges):
    walls = {}
    
    s_front = ranges[170 : 190]
    s_back  = ranges[350 : 360] + ranges[0 : 10]
    s_left  = ranges[80 : 100]
    s_right = ranges[260 : 280]
    
    s_f_nl, s_f_nr = ranges[150:160], ranges[200:210]
    s_b_nl, s_b_nr = ranges[20:30],   ranges[330:340]
    s_l_nf, s_l_nb = ranges[110:120], ranges[60:70]
    s_r_nf, s_r_nb = ranges[240:250], ranges[290:300]

    walls['Center_Front_Wall'] = "WALL" if get_avg_dist(s_front) < NEAR_WALL_MAX else "OPEN"
    walls['Center_Back_Wall']  = "WALL" if get_avg_dist(s_back) < NEAR_WALL_MAX else "OPEN"
    walls['Center_Left_Wall']  = "WALL" if get_avg_dist(s_left) < NEAR_WALL_MAX else "OPEN"
    walls['Center_Right_Wall'] = "WALL" if get_avg_dist(s_right) < NEAR_WALL_MAX else "OPEN"

    if walls['Center_Front_Wall'] == "OPEN":
        walls['Front_Neighbor_Front_Wall'] = "WALL" if (FAR_ORTHO_MIN < get_avg_dist(s_front) < FAR_ORTHO_MAX) else "OPEN"
        walls['Front_Neighbor_Left_Wall']  = "WALL" if (FAR_DIAG_MIN < get_avg_dist(s_f_nl) < FAR_DIAG_MAX) else "OPEN"
        walls['Front_Neighbor_Right_Wall'] = "WALL" if (FAR_DIAG_MIN < get_avg_dist(s_f_nr) < FAR_DIAG_MAX) else "OPEN"
    else:
        walls['Front_Neighbor_Front_Wall'] = walls['Front_Neighbor_Left_Wall'] = walls['Front_Neighbor_Right_Wall'] = "UNKNOWN"

    if walls['Center_Back_Wall'] == "OPEN":
        walls['Back_Neighbor_Back_Wall']  = "WALL" if (FAR_ORTHO_MIN < get_avg_dist(s_back) < FAR_ORTHO_MAX) else "OPEN"
        walls['Back_Neighbor_Left_Wall']  = "WALL" if (FAR_DIAG_MIN < get_avg_dist(s_b_nl) < FAR_DIAG_MAX) else "OPEN"
        walls['Back_Neighbor_Right_Wall'] = "WALL" if (FAR_DIAG_MIN < get_avg_dist(s_b_nr) < FAR_DIAG_MAX) else "OPEN"
    else:
        walls['Back_Neighbor_Back_Wall'] = walls['Back_Neighbor_Left_Wall'] = walls['Back_Neighbor_Right_Wall'] = "UNKNOWN"

    if walls['Center_Left_Wall'] == "OPEN":
        walls['Left_Neighbor_Left_Wall']  = "WALL" if (FAR_ORTHO_MIN < get_avg_dist(s_left) < FAR_ORTHO_MAX) else "OPEN"
        walls['Left_Neighbor_Front_Wall'] = "WALL" if (FAR_DIAG_MIN < get_avg_dist(s_l_nf) < FAR_DIAG_MAX) else "OPEN"
        walls['Left_Neighbor_Back_Wall']  = "WALL" if (FAR_DIAG_MIN < get_avg_dist(s_l_nb) < FAR_DIAG_MAX) else "OPEN"
    else:
        walls['Left_Neighbor_Left_Wall'] = walls['Left_Neighbor_Front_Wall'] = walls['Left_Neighbor_Back_Wall'] = "UNKNOWN"

    if walls['Center_Right_Wall'] == "OPEN":
        walls['Right_Neighbor_Right_Wall'] = "WALL" if (FAR_ORTHO_MIN < get_avg_dist(s_right) < FAR_ORTHO_MAX) else "OPEN"
        walls['Right_Neighbor_Front_Wall'] = "WALL" if (FAR_DIAG_MIN < get_avg_dist(s_r_nf) < FAR_DIAG_MAX) else "OPEN"
        walls['Right_Neighbor_Back_Wall']  = "WALL" if (FAR_DIAG_MIN < get_avg_dist(s_r_nb) < FAR_DIAG_MAX) else "OPEN"
    else:
        walls['Right_Neighbor_Right_Wall'] = walls['Right_Neighbor_Front_Wall'] = walls['Right_Neighbor_Back_Wall'] = "UNKNOWN"

    return walls

def update_walls_from_lidar():
    ranges = get_360_ranges()
    if not ranges: return

    wall_map = scan_16_walls(ranges)

    c = (current_x, current_y)
    h = current_heading

    DIRS = {
        0: {'F': (0,1),  'B': (0,-1), 'L': (-1,0), 'R': (1,0)},
        1: {'F': (1,0),  'B': (-1,0), 'L': (0,1),  'R': (0,-1)},
        2: {'F': (0,-1), 'B': (0,1),  'L': (1,0),  'R': (-1,0)},
        3: {'F': (-1,0), 'B': (1,0),  'L': (0,-1), 'R': (0,1)}
    }
    
    dF = DIRS[h]['F']
    dB = DIRS[h]['B']
    dL = DIRS[h]['L']
    dR = DIRS[h]['R']

    def add(p, d): return (p[0]+d[0], p[1]+d[1])

    mappings = {
        'Center_Front_Wall': (c, add(c, dF)),
        'Center_Back_Wall': (c, add(c, dB)),
        'Center_Left_Wall': (c, add(c, dL)),
        'Center_Right_Wall': (c, add(c, dR)),
        
        'Front_Neighbor_Front_Wall': (add(c, dF), add(add(c, dF), dF)),
        'Front_Neighbor_Left_Wall': (add(c, dF), add(add(c, dF), dL)),
        'Front_Neighbor_Right_Wall': (add(c, dF), add(add(c, dF), dR)),
        
        'Back_Neighbor_Back_Wall': (add(c, dB), add(add(c, dB), dB)),
        'Back_Neighbor_Left_Wall': (add(c, dB), add(add(c, dB), dL)),
        'Back_Neighbor_Right_Wall': (add(c, dB), add(add(c, dB), dR)),
        
        'Left_Neighbor_Left_Wall': (add(c, dL), add(add(c, dL), dL)),
        'Left_Neighbor_Front_Wall': (add(c, dL), add(add(c, dL), dF)),
        'Left_Neighbor_Back_Wall': (add(c, dL), add(add(c, dL), dB)),
        
        'Right_Neighbor_Right_Wall': (add(c, dR), add(add(c, dR), dR)),
        'Right_Neighbor_Front_Wall': (add(c, dR), add(add(c, dR), dF)),
        'Right_Neighbor_Back_Wall': (add(c, dR), add(add(c, dR), dB))
    }

    for key, (node_a, node_b) in mappings.items():
        if wall_map.get(key) == "WALL":
            wall_id = get_wall_id(node_a, node_b)
            if wall_id not in known_walls:
                known_walls.add(wall_id)


# -----------------------------------------------------------------------------
# 4. MOVEMENT & PID DRIFT PROTECTION
# -----------------------------------------------------------------------------

def move_distance(distance: float, speed: float = 2.0) -> None:
    rotation = distance / wheel_radius
    current_left = left_ps.getValue()
    current_right = right_ps.getValue()
    target_left = current_left + rotation
    target_right = current_right + rotation
    vel = abs(speed)
    left_motor.setVelocity(vel)
    right_motor.setVelocity(vel)
    left_motor.setPosition(target_left)
    right_motor.setPosition(target_right)
    while True:
        if robot.step(timestep_ms) == -1: return
        if abs(left_ps.getValue() - target_left) < 0.01 and abs(right_ps.getValue() - target_right) < 0.01:
            break
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    robot.step(timestep_ms)
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))

def normalize_angle(angle: float) -> float:
    while angle > math.pi: angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle

def turn_left_90(correct: bool = True, tolerance: float = 0.05) -> None:
    vals = compass.getValues()
    current_heading = math.atan2(vals[0], vals[1])
    target_heading = normalize_angle(current_heading + math.pi / 2)
    left_motor.setVelocity(-2.0)
    right_motor.setVelocity(2.0)
    while True:
        if robot.step(timestep_ms) == -1: break
        vals = compass.getValues()
        heading = math.atan2(vals[0], vals[1])
        error = normalize_angle(heading - target_heading)
        if abs(error) < tolerance: break
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    if correct:
        robot.step(timestep_ms)
        vals = compass.getValues()
        current_heading = math.atan2(vals[0], vals[1])
        rel_angle = normalize_angle(current_heading - initial_heading)
        multiples = [0.0, math.pi/2, math.pi, -math.pi/2, -math.pi]
        closest = min(multiples, key=lambda x: abs(normalize_angle(rel_angle - x)))
        offset = normalize_angle(rel_angle - closest)
        if abs(offset) > tolerance:
            correction_speed = 0.8 if offset > 0 else -0.8
            left_motor.setVelocity(-correction_speed)
            right_motor.setVelocity(correction_speed)
            while True:
                if robot.step(timestep_ms) == -1: break
                vals = compass.getValues()
                current_heading = math.atan2(vals[0], vals[1])
                rel_angle = normalize_angle(current_heading - initial_heading)
                offset = normalize_angle(rel_angle - closest)
                if abs(offset) < tolerance: break
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)

def turn_right_90(correct: bool = True, tolerance: float = 0.05) -> None:
    vals = compass.getValues()
    current_heading = math.atan2(vals[0], vals[1])
    target_heading = normalize_angle(current_heading - math.pi / 2)
    left_motor.setVelocity(2.0)
    right_motor.setVelocity(-2.0)
    while True:
        if robot.step(timestep_ms) == -1: break
        vals = compass.getValues()
        heading = math.atan2(vals[0], vals[1])
        error = normalize_angle(heading - target_heading)
        if abs(error) < tolerance: break
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    if correct:
        robot.step(timestep_ms)
        vals = compass.getValues()
        current_heading = math.atan2(vals[0], vals[1])
        rel_angle = normalize_angle(current_heading - initial_heading)
        multiples = [0.0, math.pi/2, math.pi, -math.pi/2, -math.pi]
        closest = min(multiples, key=lambda x: abs(normalize_angle(rel_angle - x)))
        offset = normalize_angle(rel_angle - closest)
        if abs(offset) > tolerance:
            correction_speed = 0.8 if offset > 0 else -0.8
            left_motor.setVelocity(correction_speed)
            right_motor.setVelocity(-correction_speed)
            while True:
                if robot.step(timestep_ms) == -1: break
                vals = compass.getValues()
                current_heading = math.atan2(vals[0], vals[1])
                rel_angle = normalize_angle(current_heading - initial_heading)
                offset = normalize_angle(rel_angle - closest)
                if abs(offset) < tolerance: break
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)

def move_forward_tiles(num_tiles: int, speed: float = 3.0) -> None:
    """ PID Controlled Forward Movement. Actively prevents drift while driving. """
    log_action("Moving forward 1 tile")
    steps_per_tile = steps_for_distance(tile_length, speed)
    total_steps = num_tiles * steps_per_tile
    
    vals = compass.getValues()
    current_rads = math.atan2(vals[0], vals[1])
    multiples = [0.0, math.pi/2, math.pi, -math.pi/2, -math.pi]
    locked_heading = min(multiples, key=lambda x: abs(normalize_angle(current_rads - x)))
    
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    
    for _ in range(total_steps):
        if robot.step(timestep_ms) == -1: return
        
        # 1. Compass Angular Correction
        vals = compass.getValues()
        current_rads = math.atan2(vals[0], vals[1])
        heading_error = normalize_angle(locked_heading - current_rads)
        
        # 2. LiDAR Lateral Wall Correction
        lateral_error = 0.0
        ranges = get_360_ranges()
        if ranges:
            left_slice = [r for r in ranges[88:92] if not math.isinf(r)]
            right_slice = [r for r in ranges[268:272] if not math.isinf(r)]
            
            left_dist = min(left_slice) if left_slice else float('inf')
            right_dist = min(right_slice) if right_slice else float('inf')
            
            l_wall = left_dist < 0.18
            r_wall = right_dist < 0.18
            
            if l_wall and r_wall:
                lateral_error = left_dist - right_dist
            elif l_wall:
                lateral_error = (left_dist - 0.125) * 2.0
            elif r_wall:
                lateral_error = (0.125 - right_dist) * 2.0
        
        # Combine Errors (Steer away from obstacles and maintain angle)
        correction = (heading_error * 3.0) + (lateral_error * 2.0)
        correction = max(min(correction, speed - 0.5), -(speed - 0.5))
        
        left_motor.setVelocity(speed - correction)
        right_motor.setVelocity(speed + correction)
        
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    robot.step(timestep_ms)

def adjust_to_wall(target_distance: float = 0.125, tolerance: float = 0.02,
                   check_wall: bool = True, wall_threshold: float = 0.3) -> None:
    log_action("Adjusting distance to front wall")
    robot.step(timestep_ms)
    if check_wall:
        ranges = get_360_ranges()
        if ranges:
            front_slice = [r for r in ranges[178:183] if not math.isinf(r)]
            front = min(front_slice) if front_slice else float('inf')
            if front > wall_threshold: return
    
    max_speed = 1.5  
    min_speed = 0.2
    while True:
        if robot.step(timestep_ms) == -1: break
        ranges = get_360_ranges()
        if not ranges: continue
        
        front_slice = [r for r in ranges[178:183] if not math.isinf(r)]
        front = min(front_slice) if front_slice else float('inf')
        
        if math.isinf(front): break
        error = front - target_distance
        if abs(error) <= tolerance: break
        speed = max(min_speed, min(max_speed, abs(error) * 5))
        if error > 0:
            left_motor.setVelocity(speed)
            right_motor.setVelocity(speed)
        else:
            left_motor.setVelocity(-speed)
            right_motor.setVelocity(-speed)
            
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

#----------------------------------------------
# Navigation Primitives
#----------------------------------------------
def turn_to_heading(target_h: int):
    global current_heading
    log_action(f"Turning from {current_heading} to {target_h}")
    diff = (target_h - current_heading) % 4
    if diff == 1: turn_right_90() 
    elif diff == 2: turn_right_90(); turn_right_90()
    elif diff == 3: turn_left_90()  
    current_heading = target_h

def check_wall_ahead(threshold: float = 0.18) -> bool:
    robot.step(timestep_ms)
    ranges = get_360_ranges()
    if ranges:
        span = 15 
        front_rays = ranges[180 - span : 180 + span]
        valid_rays = [r for r in front_rays if not math.isinf(r)]
        if valid_rays:
            if min(valid_rays) < threshold: return True
    return False

def navigate_to_target(target_x, target_y):
    global current_x, current_y, current_heading
    
    robot.step(timestep_ms)
    update_walls_from_lidar()
    
    while (current_x, current_y) != (target_x, target_y):
        path = get_shortest_path_bfs((current_x, current_y), (target_x, target_y))
        if not path or len(path) < 2: 
            log_action("NAV ERROR: No viable path found via BFS.")
            return False
            
        next_node = path[1]
        target_h = get_target_heading((current_x, current_y), next_node)
        
        if target_h != current_heading:
            turn_to_heading(target_h)
            robot.step(timestep_ms)
            update_walls_from_lidar()
            
        if check_wall_ahead():
            log_action("Obstacle detected! Adjusting to wall.")
            adjust_to_wall(target_distance=0.125)
            robot.step(timestep_ms)
            update_walls_from_lidar()
            continue 
            
        move_forward_tiles(1)
        current_x, current_y = next_node
            
        robot.step(timestep_ms)
        update_walls_from_lidar()
        
    return True


# -----------------------------------------------------------------------------
# 5. MAIN EXECUTION
# -----------------------------------------------------------------------------

current_x = 0
current_y = 0
current_heading = 0 
known_walls = set() 

def detect_final_wall_color() -> str:
    cam.enable(timestep_ms)
    robot.step(timestep_ms)
    robot.step(timestep_ms) 
    raw = cam.getImage()
    cam.disable()
    if not raw: return "Unknown"
    
    img = np.frombuffer(raw, np.uint8).reshape((cam.getHeight(), cam.getWidth(), 4))
    h, w = img.shape[:2]
    center_patch = img[h//2-10:h//2+10, w//2-10:w//2+10]
    mean_color = cv2.mean(center_patch)
    b, g, r = mean_color[0], mean_color[1], mean_color[2]
    
    if max(r, g, b) < 60: return "Black"
    elif r > g and r > b: return "Red"
    elif g > r and g > b: return "Green"
    elif b > r and b > g: return "Blue"
    return "Unknown"

def get_wall_id(node_a, node_b): return tuple(sorted([node_a, node_b]))

def get_shortest_path_bfs(start, target, max_extent=25):
    if start == target: return [start]
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current_node, path = queue.popleft()
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_node = (current_node[0] + dx, current_node[1] + dy)
            if not (-max_extent <= next_node[0] <= max_extent and -max_extent <= next_node[1] <= max_extent): continue
            
            wall_boundary = get_wall_id(current_node, next_node)
            if wall_boundary in known_walls: continue
            if next_node in visited: continue
                
            new_path = path + [next_node]
            if next_node == target: return new_path
                
            visited.add(next_node)
            queue.append((next_node, new_path))
    return None

def get_target_heading(current_node, next_node):
    dx = next_node[0] - current_node[0]
    dy = next_node[1] - current_node[1]
    if dy == 1: return 0  
    if dx == 1: return 1  
    if dy == -1: return 2 
    if dx == -1: return 3 
    return 0

def main() -> None:
    global current_x, current_y
    
    try:
        robot.step(timestep_ms) 
        update_walls_from_lidar() 
        
        while(not check_wall_ahead()):
            move_forward_tiles(1)
            current_y += 1 
            if check_wall_ahead(): break
            robot.step(timestep_ms)
            update_walls_from_lidar()
        
        adjust_to_wall()
        robot.step(timestep_ms)
        update_walls_from_lidar()
        
        while robot.step(timestep_ms) != -1:
            x, y, tag_id, binary_str = scan_aruco_tag()
            if tag_id is not None:
                log_action(f"Scanned Code: Target ({x}, {y})")
                print(f"Scanned Code: Target ({x}, {y})")
                if x == current_x and y == current_y: break
                success = navigate_to_target(x, y)
                if not success: break 
            else: break
                
        color = detect_final_wall_color()
        print(f"Final Wall Color Scanned: {color}")
        send_message(color)

        print("\n--- Final Walls Mapped ---")
        for wall in sorted(list(known_walls)): print(f"Wall between {wall[0]} and {wall[1]}")
        
    except Exception as e:
        print(f"\n[EXCEPTION CAUGHT]: {e}")
    finally:
        dump_crash_log()

if __name__ == '__main__':
    main()