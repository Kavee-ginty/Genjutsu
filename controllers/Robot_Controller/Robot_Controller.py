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
    return result == 1


# -----------------------------------------------------------------------------
# 2. ARUCO TAG DETECTION
# -----------------------------------------------------------------------------

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

def scan_aruco_tag(wall_threshold: float = 0.3, scan_distance: float = 0.2) -> tuple:
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
                move_distance(-scan_distance)
                result = detect_once()
                move_distance(scan_distance)
                
                if result[2] is not None: 
                    for _ in range(turns):
                        turn_left_90(correct=False)
                    lidar.disable()
                    return result

        turn_right_90(correct=False)
        turns += 1
        
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
                lidar.disable()
                return
    
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
    turn_left_90(correct=True)
    if check_wall_ahead(threshold=wall_threshold):
        adjust_to_wall()
        turn_right_90(correct=True)
        return

    turn_right_90(correct=True)
    turn_right_90(correct=True)
    if check_wall_ahead(threshold=wall_threshold):
        adjust_to_wall()
        turn_left_90(correct=True)
        return
        
    turn_left_90(correct=True)

#----------------------------------------------
# Navigation Primitives
#----------------------------------------------
def turn_to_heading(target_h: int):
    global current_heading
    diff = (target_h - current_heading) % 4
    
    if diff == 1:
        turn_left_90()
    elif diff == 2:
        turn_left_90()
        turn_left_90()
    elif diff == 3:
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
            if min_dist < threshold:
                return True
    return False

def navigate_to_target(target_x, target_y):
    global current_x, current_y, current_heading
    straight_tiles_count = 0
    
    while (current_x, current_y) != (target_x, target_y):
        path = get_shortest_path_astar((current_x, current_y), (target_x, target_y))
        
        if not path or len(path) < 2:
            return False
            
        next_node = path[1]
        target_h = get_target_heading((current_x, current_y), next_node)
        
        if target_h != current_heading:
            turn_to_heading(target_h)
            straight_tiles_count = 0 
            
        if check_wall_ahead():
            adjust_to_wall()
            blocked_boundary = get_wall_id((current_x, current_y), next_node)
            known_walls.add(blocked_boundary)
            straight_tiles_count = 0 
            continue 
            
        move_forward_tiles(1)
        current_x, current_y = next_node
        straight_tiles_count += 1
        
        if straight_tiles_count >= 4:
            adjust_to_side_wall()
            straight_tiles_count = 0 
            
    print(f"Reached coordinate: ({target_x}, {target_y})")
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
    
    while(not check_wall_ahead()):
        move_forward_tiles(1)
        current_y += 1 
    
    adjust_to_wall()
    
    while robot.step(timestep_ms) != -1:
        
        x, y, tag_id, binary_str = scan_aruco_tag()
        
        if tag_id is not None:
            print(f"Decoded (X, Y) Coordinates: ({x}, {y})")
            
            if x == current_x and y == current_y:
                break
                
            success = navigate_to_target(x, y)
            if not success:
                break 
                
        else:
            break
            
    color = detect_final_wall_color()
    print(f"Final Wall Color: {color}")
    send_message(color)


if __name__ == '__main__':
    main()