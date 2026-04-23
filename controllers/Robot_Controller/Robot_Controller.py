# Improved version of the e‑puck controller.
import math
import cv2
import numpy as np
from controller import Robot
import collections
import heapq

# -----------------------------------------------------------------------------
# 1. INITIALIZATION & CONSTANTS
# -----------------------------------------------------------------------------

robot = Robot()
timestep_ms = int(robot.getBasicTimeStep())
timestep = timestep_ms / 1000.0

tile_length = 0.25  
wheel_radius = 0.0205  

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
lidar.enablePointCloud()
lidar.disable()  
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
    """
    Sends a string message to the supervisor.
    """
    binary_data = message_string.encode('utf-8')
    result = emitter.send(binary_data)

    if result == 1:
        print(f"E-puck: '{message_string}' sent successfully!")
        return True
    else:
        return False


# -----------------------------------------------------------------------------
# 2. ARUCO TAG DETECTION
# -----------------------------------------------------------------------------

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

def scan_aruco_tag(wall_threshold: float = 0.3, scan_distance: float = 0.2) -> tuple:
    print("\n[SCANNER] Initiating 360-degree room scan for ArUco tags...")
    def detect_once() -> tuple:
        cam.enable(timestep_ms)
        robot.step(timestep_ms)
        raw = cam.getImage()
        cam.disable()
        if not raw:
            return None, None, None, None
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

    lidar.enable(timestep_ms)
    robot.step(timestep_ms)
    turns = 0  
    
    for attempt in range(4):
        if attempt > 0:
            robot.step(timestep_ms)
        ranges = lidar.getRangeImage()
        if ranges:
            center = len(ranges) // 2
            front_dist = ranges[center]
            
            if not math.isinf(front_dist) and front_dist < wall_threshold:
                print(f"[SCANNER] Wall found at {front_dist:.3f}m. Backing up to take photo...")
                move_distance(-scan_distance)
                result = detect_once()
                move_distance(scan_distance)
                
                if result[2] is not None: 
                    print(f"[SCANNER] SUCCESS! Tag {result[2]} found.")
                    for _ in range(turns):
                        turn_left_90(correct=False)
                    lidar.disable()
                    return result
                else:
                    print("[SCANNER] Wall is blank. No tag detected.")

        print("[SCANNER] Turning right to check next wall...")
        turn_right_90(correct=False)
        turns += 1
        
    print("[SCANNER] All 4 sides checked. No tags found.")
    for _ in range(turns):
        turn_left_90(correct=False)
    lidar.disable()
    
    return None, None, None, None

# -----------------------------------------------------------------------------
# 3. MOVEMENT PRIMITIVES
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
        if robot.step(timestep_ms) == -1:
            return
        if abs(left_ps.getValue() - target_left) < 0.01 and abs(right_ps.getValue() - target_right) < 0.01:
            break
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    robot.step(timestep_ms)
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))

def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def turn_left_90(correct: bool = True, tolerance: float = 0.05) -> None:
    vals = compass.getValues()
    current_heading = math.atan2(vals[0], vals[1])
    target_heading = normalize_angle(current_heading + math.pi / 2)
    left_motor.setVelocity(-2.0)
    right_motor.setVelocity(2.0)
    while True:
        if robot.step(timestep_ms) == -1:
            break
        vals = compass.getValues()
        heading = math.atan2(vals[0], vals[1])
        error = normalize_angle(heading - target_heading)
        if abs(error) < tolerance:
            break
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
                if robot.step(timestep_ms) == -1:
                    break
                vals = compass.getValues()
                current_heading = math.atan2(vals[0], vals[1])
                rel_angle = normalize_angle(current_heading - initial_heading)
                offset = normalize_angle(rel_angle - closest)
                if abs(offset) < tolerance:
                    break
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)

def turn_right_90(correct: bool = True, tolerance: float = 0.05) -> None:
    vals = compass.getValues()
    current_heading = math.atan2(vals[0], vals[1])
    target_heading = normalize_angle(current_heading - math.pi / 2)
    left_motor.setVelocity(2.0)
    right_motor.setVelocity(-2.0)
    while True:
        if robot.step(timestep_ms) == -1:
            break
        vals = compass.getValues()
        heading = math.atan2(vals[0], vals[1])
        error = normalize_angle(heading - target_heading)
        if abs(error) < tolerance:
            break
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
                if robot.step(timestep_ms) == -1:
                    break
                vals = compass.getValues()
                current_heading = math.atan2(vals[0], vals[1])
                rel_angle = normalize_angle(current_heading - initial_heading)
                offset = normalize_angle(rel_angle - closest)
                if abs(offset) < tolerance:
                    break
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)

def move_forward_tiles(num_tiles: int, speed: float = 3.0) -> None:
    steps_per_tile = steps_for_distance(tile_length, speed)
    for _ in range(num_tiles):
        left_motor.setVelocity(speed)
        right_motor.setVelocity(speed)
        for _ in range(steps_per_tile):
            if robot.step(timestep_ms) == -1:
                return
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        robot.step(timestep_ms)

def adjust_to_wall(target_distance: float = 0.128, tolerance: float = 0.02,
                   check_wall: bool = True, wall_threshold: float = 0.3) -> None:
    lidar.enable(timestep_ms)
    robot.step(timestep_ms)
    if check_wall:
        ranges = lidar.getRangeImage()
        if ranges:
            center = len(ranges) // 2
            front = ranges[center]
            if math.isinf(front) or front > wall_threshold:
                print(f"   -> [ADJUST] No wall close enough to adjust to (Dist: {front:.3f}m). Skipping.")
                lidar.disable()
                return
    
    print(f"   -> [ADJUST] Squaring up to wall (Target: {target_distance}m)...")
    max_speed = 1.5  
    min_speed = 0.2
    while True:
        if robot.step(timestep_ms) == -1:
            break
        values = lidar.getRangeImage()
        if not values:
            continue
        center = len(values) // 2
        front = values[center]
        if math.isinf(front):
            break
        error = front - target_distance
        if abs(error) <= tolerance:
            print(f"   -> [ADJUST] Centered successfully. Error is {error:.3f}m.")
            break
        speed = max(min_speed, min(max_speed, abs(error) * 5))
        if error > 0:
            left_motor.setVelocity(speed)
            right_motor.setVelocity(speed)
        else:
            left_motor.setVelocity(-speed)
            right_motor.setVelocity(-speed)
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    lidar.disable()

def adjust_to_side_wall(wall_threshold: float = 0.25) -> None:
    print("\n[DRIFT FIX] Initiating side-wall check to re-center Y-axis...")
    print("   -> Turning Left 90 deg...")
    turn_left_90(correct=True)
    
    if check_wall_ahead(threshold=wall_threshold):
        print("   -> Left wall detected. Adjusting to it.")
        adjust_to_wall()
        print("   -> Turning Right 90 deg back to forward path.")
        turn_right_90(correct=True)
        return

    print("   -> No Left wall. Turning 180 deg to check Right wall...")
    turn_right_90(correct=True)
    turn_right_90(correct=True)
    
    if check_wall_ahead(threshold=wall_threshold):
        print("   -> Right wall detected. Adjusting to it.")
        adjust_to_wall()
        print("   -> Turning Left 90 deg back to forward path.")
        turn_left_90(correct=True)
        return
        
    print("   -> Open intersection. No walls on either side. Resuming.")
    turn_left_90(correct=True)

#----------------------------------------------
# Navigation Primitives
#----------------------------------------------
def turn_to_heading(target_h: int):
    global current_heading
    diff = (target_h - current_heading) % 4
    
    if diff == 1:
        print("   -> [TURN] Action: Turn Left 90°")
        turn_left_90()
    elif diff == 2:
        print("   -> [TURN] Action: Turn Left 180°")
        turn_left_90()
        turn_left_90()
    elif diff == 3:
        print("   -> [TURN] Action: Turn Right 90°")
        turn_right_90()
        
    current_heading = target_h

def check_wall_ahead(threshold: float = 0.18) -> bool:
    lidar.enable(timestep_ms)
    robot.step(timestep_ms)
    ranges = lidar.getRangeImage()
    lidar.disable()
    
    if ranges:
        center = len(ranges) // 2
        span = max(1, len(ranges) // 15) 
        front_rays = ranges[center - span : center + span]
        
        valid_rays = [r for r in front_rays if not math.isinf(r)]
        if valid_rays:
            min_dist = min(valid_rays)
            print(f"   -> [LiDAR RAW] Closest object in front cone is {min_dist:.3f}m away.")
            if min_dist < threshold:
                print(f"   -> [LiDAR ALERT] {min_dist:.3f}m is < {threshold}m threshold. WALL CONFIRMED.")
                return True
            else:
                print(f"   -> [LiDAR SAFE] {min_dist:.3f}m means path is clear.")
    return False

def navigate_to_target(target_x, target_y):
    global current_x, current_y, current_heading
    straight_tiles_count = 0
    print(f"\n========== NEW NAVIGATION GOAL: ({target_x}, {target_y}) ==========")
    
    while (current_x, current_y) != (target_x, target_y):
        print(f"\n[NAV-LOOP] Pos: ({current_x}, {current_y}) | Heading: {current_heading}")
        path = get_shortest_path_astar((current_x, current_y), (target_x, target_y))
        
        if not path or len(path) < 2:
            print(f"[FATAL ERROR] A* cannot find a path! Known walls must be completely boxing us in.")
            print(f"   -> Known walls memory dump: {known_walls}")
            return False
            
        print(f"   -> [A* CALC] Shortest path looks like: {path}")
        next_node = path[1]
        target_h = get_target_heading((current_x, current_y), next_node)
        
        if target_h != current_heading:
            print(f"   -> [ACTION] Need to change heading from {current_heading} to {target_h}")
            turn_to_heading(target_h)
            straight_tiles_count = 0 
            
        print(f"   -> [ACTION] Scanning forward path to {next_node}...")
        if check_wall_ahead():
            print(f"   -> [OBSTACLE] Unexpected wall detected between {(current_x, current_y)} and {next_node}!")
            
            print("   -> [ACTION] Squaring up to this wall to fix drift before recalculating...")
            adjust_to_wall()
            
            blocked_boundary = get_wall_id((current_x, current_y), next_node)
            known_walls.add(blocked_boundary)
            print(f"   -> [MAPPING] Wall {blocked_boundary} saved to memory. Restarting loop to replan.")
            straight_tiles_count = 0 
            continue 
            
        print(f"   -> [ACTION] Path is physically clear. Moving Forward 1 tile to {next_node}.")
        move_forward_tiles(1)
        current_x, current_y = next_node
        straight_tiles_count += 1
        
        if straight_tiles_count >= 4:
            print(f"   -> [TRIGGER] Moved 4 straight tiles. Activating side-wall drift check.")
            adjust_to_side_wall()
            straight_tiles_count = 0 
            
    print(f"========== ARRIVED AT GOAL: ({target_x}, {target_y}) ==========\n")
    return True

# -----------------------------------------------------------------------------
# 4. MAIN EXECUTION
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

def get_wall_id(node_a, node_b):
    return tuple(sorted([node_a, node_b]))

def get_heuristic(node, target):
    return abs(node[0] - target[0]) + abs(node[1] - target[1])

def get_shortest_path_astar(start, target, grid_size=12):
    queue = [(get_heuristic(start, target), 0, start, [start])]
    visited_costs = {start: 0}
    
    while queue:
        f, g, current_node, path = heapq.heappop(queue)
        if current_node == target:
            return path
            
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_node = (current_node[0] + dx, current_node[1] + dy)
            if not (0 <= next_node[0] < grid_size and 0 <= next_node[1] < grid_size):
                continue
                
            wall_boundary = get_wall_id(current_node, next_node)
            if wall_boundary in known_walls:
                continue
                
            new_g = g + 1
            if next_node not in visited_costs or new_g < visited_costs[next_node]:
                visited_costs[next_node] = new_g
                new_f = new_g + get_heuristic(next_node, target)
                heapq.heappush(queue, (new_f, new_g, next_node, path + [next_node]))
                
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
    
    robot.step(timestep_ms) 
    
    print("========== BOOT SEQUENCE ==========")
    print("Moving forward to first wall...")
    while(not check_wall_ahead()):
        move_forward_tiles(1)
        current_y += 1 
    
    print("Initial blind run complete. Squaring up to starting wall...")
    adjust_to_wall()
    
    while robot.step(timestep_ms) != -1:
        
        x, y, tag_id, binary_str = scan_aruco_tag()
        
        if tag_id is not None:
            print(f"--- VALID TAG DETECTED ---")
            print(f"Decimal ID: {tag_id} | Binary: {binary_str}")
            print(f"Decoded Target Coordinate: X={x}, Y={y}")
            
            if x == current_x and y == current_y:
                print("[SUCCESS] Tag coordinate matches current location. Ending navigation.")
                break
                
            success = navigate_to_target(x, y)
            if not success:
                break 
                
        else:
            print("\n[FINISH] No tags found in room. Reached final location.")
            break
            
    print("\nAttempting to read final wall color...")
    color = detect_final_wall_color()
    print(f"Final Wall Color Detected: {color}")
    send_message(color)


if __name__ == '__main__':
    main()