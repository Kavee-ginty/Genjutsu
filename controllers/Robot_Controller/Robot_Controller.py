# E-Puck Maze Controller — 8-Wall Orthogonal Lidar Mapping
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

tile_length  = 0.25
wheel_radius = 0.0205
turn_counter = 0

emitter     = robot.getDevice('emitter_2')
cam         = robot.getDevice('camera')
gps         = robot.getDevice('gps')
compass     = robot.getDevice('compass')
lidar       = robot.getDevice('lidar')
left_motor  = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_ps  = left_motor.getPositionSensor()
right_ps = right_motor.getPositionSensor()
left_ps.enable(timestep_ms)
right_ps.enable(timestep_ms)

gps.enable(timestep_ms)
compass.enable(timestep_ms)

# 360-ray lidar — enabled ONCE at startup, stays on for the whole run
lidar.enable(timestep_ms)
lidar.enablePointCloud()

cam.disable()

robot.step(timestep_ms)
initial_heading = 0.0
if compass:
    initial_vals    = compass.getValues()
    initial_heading = math.atan2(initial_vals[0], initial_vals[1])

start_gps_x = 1.380
start_gps_y = -1.388

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# --- Lidar wall-detection thresholds ---
NEAR_WALL_MAX = 0.20   # wall is right next to the robot (<0.20 m)
FAR_WALL_MIN  = 0.30   # far side of next tile
FAR_WALL_MAX  = 0.45

# -----------------------------------------------------------------------------
# 2. 8-WALL LIDAR FUNCTIONS  (360-ray, 6.28 rad FOV)
# -----------------------------------------------------------------------------

def get_avg_dist(sector):
    """Average of finite, positive readings in a lidar sector."""
    valid = [r for r in sector if r != float('inf') and r > 0]
    return sum(valid) / len(valid) if valid else float('inf')


def scan_8_walls(ranges, f, N):
    """
    Slice the 360-ray range image into 4 orthogonal sectors and return an
    8-entry dict describing walls in the current tile and one look-ahead tile
    in each cardinal direction.

    Sector centres (robot-relative, counter-clockwise from front):
        front  -> index ~180   (ray pointing straight ahead)
        left   -> index ~90
        back   -> index ~0/360 (wrap-around)
        right  -> index ~270

    Returns keys: front, back, left, right,
                  front_front, back_back, left_left, right_right
    Each value is "WALL", "OPEN", or "UNKNOWN".
    """
    walls = {}

    s_front = ranges[int(170*f) : int(190*f)]
    s_back  = ranges[int(350*f):N] + ranges[0:int(10*f)]
    s_left  = ranges[int(80*f)  : int(100*f)]
    s_right = ranges[int(260*f) : int(280*f)]

    # FRONT
    d_front = get_avg_dist(s_front)
    if d_front < NEAR_WALL_MAX:
        walls['front']       = "WALL"
        walls['front_front'] = "UNKNOWN"
    else:
        walls['front']       = "OPEN"
        walls['front_front'] = "WALL" if FAR_WALL_MIN < d_front < FAR_WALL_MAX else "OPEN"

    # BACK
    d_back = get_avg_dist(s_back)
    if d_back < NEAR_WALL_MAX:
        walls['back']      = "WALL"
        walls['back_back'] = "UNKNOWN"
    else:
        walls['back']      = "OPEN"
        walls['back_back'] = "WALL" if FAR_WALL_MIN < d_back < FAR_WALL_MAX else "OPEN"

    # LEFT
    d_left = get_avg_dist(s_left)
    if d_left < NEAR_WALL_MAX:
        walls['left']      = "WALL"
        walls['left_left'] = "UNKNOWN"
    else:
        walls['left']      = "OPEN"
        walls['left_left'] = "WALL" if FAR_WALL_MIN < d_left < FAR_WALL_MAX else "OPEN"

    # RIGHT
    d_right = get_avg_dist(s_right)
    if d_right < NEAR_WALL_MAX:
        walls['right']       = "WALL"
        walls['right_right'] = "UNKNOWN"
    else:
        walls['right']       = "OPEN"
        walls['right_right'] = "WALL" if FAR_WALL_MIN < d_right < FAR_WALL_MAX else "OPEN"

    return walls


def get_wall_map():
    """Return the current 8-wall map from a single lidar snapshot."""
    robot.step(timestep_ms)
    ranges = lidar.getRangeImage()
    if not ranges:
        return {}
    N = len(ranges)
    f = N / 360
    return scan_8_walls(ranges, f, N)


def scan_and_register_walls():
    """
    Read the 8-wall map and immediately add confirmed walls to known_walls.
    This is called proactively after every move so BFS always has the most
    up-to-date picture of the maze.

    Heading encoding used throughout:
        0 = +Y (North / forward at start)
        1 = +X (East  / right at start)
        2 = -Y (South / backward at start)
        3 = -X (West  / left at start)
    """
    wall_map = get_wall_map()
    if not wall_map:
        return

    # Map each sensor direction to the (dx, dy) step it represents.
    # current_heading tells us where the robot's "front" points on the grid.
    heading_to_delta = {
        0: (0,  1),   # North
        1: (1,  0),   # East
        2: (0, -1),   # South
        3: (-1, 0),   # West
    }

    # Relative sensor directions expressed as offsets from current_heading (mod 4)
    sensor_heading_offsets = {
        'front' :  0,
        'right' :  3,   # robot-right = heading rotated -90 deg = (heading+3)%4
        'back'  :  2,
        'left'  :  1,   # robot-left  = heading rotated +90 deg = (heading+1)%4
    }

    def register_if_wall(sensor_key, tile_delta_steps):
        """
        sensor_key        : one of 'front','back','left','right'
        tile_delta_steps  : how many tiles away to register (1 = adjacent, 2 = look-ahead)
        """
        if wall_map.get(sensor_key) != "WALL":
            return
        abs_heading = (current_heading + sensor_heading_offsets[sensor_key]) % 4
        dx, dy = heading_to_delta[abs_heading]
        neighbor = (current_x + dx * tile_delta_steps,
                    current_y + dy * tile_delta_steps)
        here     = (current_x + dx * (tile_delta_steps - 1),
                    current_y + dy * (tile_delta_steps - 1))
        wall_id  = get_wall_id(here, neighbor)
        known_walls.add(wall_id)

    # Register immediate (adjacent) walls
    for direction in ('front', 'back', 'left', 'right'):
        register_if_wall(direction, 1)

    # Register look-ahead walls (2 tiles away) — only if the near tile is OPEN
    look_ahead_map = {
        'front_front' : 'front',
        'back_back'   : 'back',
        'left_left'   : 'left',
        'right_right' : 'right',
    }
    for far_key, near_key in look_ahead_map.items():
        if wall_map.get(near_key) == "OPEN" and wall_map.get(far_key) == "WALL":
            register_if_wall(near_key, 2)

# -----------------------------------------------------------------------------
# 3. HELPER UTILITIES
# -----------------------------------------------------------------------------

def steps_for_distance(distance, angular_speed):
    linear_speed = abs(angular_speed) * wheel_radius
    if linear_speed <= 0.0:
        return 0
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


def get_heuristic(node, target):
    return abs(node[0] - target[0]) + abs(node[1] - target[1])

# -----------------------------------------------------------------------------
# 4. ARUCO TAG DETECTION
# -----------------------------------------------------------------------------

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector   = cv2.aruco.ArucoDetector(aruco_dict, parameters)


def scan_aruco_tag(scan_distance=0.2):
    """
    Rotate up to 4 times looking for a wall ahead (using 8-wall scan).
    When a front wall is found, back up, grab a camera frame, then return.
    """
    def detect_once():
        cam.enable(timestep_ms)
        robot.step(timestep_ms)
        raw = cam.getImage()
        cam.disable()
        if not raw:
            return None, None, None, None
        img   = np.frombuffer(raw, np.uint8).reshape((cam.getHeight(), cam.getWidth(), 4))
        gray  = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            tag_id     = int(ids[0][0])
            binary_str = format(tag_id, '08b')
            x = (tag_id >> 4) & 0x0F
            y =  tag_id       & 0x0F
            return x, y, tag_id, binary_str
        return None, None, None, None

    turns = 0
    for attempt in range(4):
        # Use the always-on 360 lidar via get_wall_map()
        wall_map = get_wall_map()
        if wall_map.get('front') == "WALL":
            move_distance(-scan_distance)
            result = detect_once()
            move_distance(scan_distance)
            if result[2] is not None:
                # Undo any turns we did before finding the tag
                for _ in range(turns):
                    turn_left_90(correct=False)
                return result

        turn_right_90(correct=False)
        turns += 1

    # Restore original orientation
    for _ in range(turns):
        turn_left_90(correct=False)

    return None, None, None, None

# -----------------------------------------------------------------------------
# 5. MOVEMENT PRIMITIVES
# -----------------------------------------------------------------------------

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
        if robot.step(timestep_ms) == -1:
            return
        if (abs(left_ps.getValue()  - target_left)  < 0.01 and
                abs(right_ps.getValue() - target_right) < 0.01):
            break
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    robot.step(timestep_ms)
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))


def _snap_heading(tolerance=0.05):
    """Correct tiny orientation errors after a turn."""
    vals        = compass.getValues()
    cur_heading = math.atan2(vals[0], vals[1])
    rel_angle   = normalize_angle(cur_heading - initial_heading)
    multiples   = [0.0, math.pi/2, math.pi, -math.pi/2, -math.pi]
    closest     = min(multiples, key=lambda x: abs(normalize_angle(rel_angle - x)))
    offset      = normalize_angle(rel_angle - closest)
    if abs(offset) > tolerance:
        spd = 0.8 if offset > 0 else -0.8
        left_motor.setVelocity(-spd)
        right_motor.setVelocity( spd)
        while True:
            if robot.step(timestep_ms) == -1:
                break
            vals      = compass.getValues()
            cur_h     = math.atan2(vals[0], vals[1])
            rel_angle = normalize_angle(cur_h - initial_heading)
            offset    = normalize_angle(rel_angle - closest)
            if abs(offset) < tolerance:
                break
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)


def turn_left_90(correct=True, tolerance=0.05):
    global turn_counter
    turn_counter += 1
    if turn_counter == 5:
        print("5th turn, Performing wall adjustment to correct drift...")
        adjust_to_wall()
        turn_counter = 0
    vals           = compass.getValues()
    cur_h          = math.atan2(vals[0], vals[1])
    target_heading = normalize_angle(cur_h + math.pi / 2)
    left_motor.setVelocity(-2.0)
    right_motor.setVelocity( 2.0)
    while True:
        if robot.step(timestep_ms) == -1:
            break
        vals  = compass.getValues()
        h     = math.atan2(vals[0], vals[1])
        error = normalize_angle(h - target_heading)
        if abs(error) < tolerance:
            break
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    if correct:
        robot.step(timestep_ms)
        _snap_heading(tolerance)
    

def turn_right_90(correct=True, tolerance=0.05):
    global turn_counter
    turn_counter += 1
    if turn_counter == 5:
        print("5th turn, Performing wall adjustment to correct drift...")
        adjust_to_wall()
        turn_counter = 0
    vals           = compass.getValues()
    cur_h          = math.atan2(vals[0], vals[1])
    target_heading = normalize_angle(cur_h - math.pi / 2)
    left_motor.setVelocity( 2.0)
    right_motor.setVelocity(-2.0)
    while True:
        if robot.step(timestep_ms) == -1:
            break
        vals  = compass.getValues()
        h     = math.atan2(vals[0], vals[1])
        error = normalize_angle(h - target_heading)
        if abs(error) < tolerance:
            break
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    if correct:
        robot.step(timestep_ms)
        _snap_heading(tolerance)


def move_forward_tiles(num_tiles, speed=3.0):
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


def adjust_to_wall(target_distance=0.128, tolerance=0.02, wall_threshold=0.3):
    """
    Drive forward/backward until the front lidar sector reads target_distance.
    Uses the always-on 360 lidar — no enable/disable needed.
    """

    # --- Adjust to front wall as before ---
    ranges = lidar.getRangeImage()
    if ranges:
        N = len(ranges)
        f = N / 360
        s_front = ranges[int(170*f) : int(190*f)]
        d = get_avg_dist(s_front)
        if math.isinf(d) or d > wall_threshold:
            return   # nothing to align to
        if d >= 0.09 and d <= 0.17:
            return   # avoid over-correction

    max_speed = 1.5
    min_speed = 0.2
    while True:
        if robot.step(timestep_ms) == -1:
            break
        ranges = lidar.getRangeImage()
        if not ranges:
            continue
        N = len(ranges)
        f = N / 360
        s_front = ranges[int(170*f) : int(190*f)]
        front   = get_avg_dist(s_front)
        if math.isinf(front):
            break
        error = front - target_distance
        if abs(error) <= tolerance:
            break
        speed = max(min_speed, min(max_speed, abs(error) * 5))
        if error > 0:
            left_motor.setVelocity( speed)
            right_motor.setVelocity( speed)
        else:
            left_motor.setVelocity(-speed)
            right_motor.setVelocity(-speed)
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    print(f"Adjusted to wall: distance={front:.3f}, error={error:.3f}")

    # --- After adjusting to front wall, check for side wall ---
    robot.step(timestep_ms)
    ranges = lidar.getRangeImage()
    if not ranges:
        return
    N = len(ranges)
    f = N / 360
    s_left = ranges[int(80*f) : int(100*f)]
    s_right = ranges[int(260*f) : int(280*f)]
    d_left = get_avg_dist(s_left)
    d_right = get_avg_dist(s_right)

    # Save current heading
    vals = compass.getValues()
    cur_heading = math.atan2(vals[0], vals[1])

    # Prefer left wall if both present
    side_aligned = False
    if not math.isinf(d_left) and d_left < wall_threshold:
        turn_left_90(correct=False)
        # Align to left wall
        _adjust_to_side(target_distance, tolerance, wall_threshold)
        turn_right_90(correct=False)
        side_aligned = True
    elif not math.isinf(d_right) and d_right < wall_threshold:
        turn_right_90(correct=False)
        # Align to right wall
        _adjust_to_side(target_distance, tolerance, wall_threshold)
        turn_left_90(correct=False)
        side_aligned = True

    if side_aligned:
        print("Adjusted to side wall and returned to original heading.")


# Helper for side wall adjustment (same logic as front, but for current heading)
def _adjust_to_side(target_distance=0.128, tolerance=0.02, wall_threshold=0.3):
    # This function assumes robot is already facing the side wall
    max_speed = 1.5
    min_speed = 0.2
    while True:
        if robot.step(timestep_ms) == -1:
            break
        ranges = lidar.getRangeImage()
        if not ranges:
            continue
        N = len(ranges)
        f = N / 360
        s_front = ranges[int(170*f) : int(190*f)]
        front   = get_avg_dist(s_front)
        if math.isinf(front):
            break
        error = front - target_distance
        if abs(error) <= tolerance:
            break
        speed = max(min_speed, min(max_speed, abs(error) * 5))
        if error > 0:
            left_motor.setVelocity( speed)
            right_motor.setVelocity( speed)
        else:
            left_motor.setVelocity(-speed)
            right_motor.setVelocity(-speed)
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    print(f"Adjusted to side wall: distance={front:.3f}, error={error:.3f}")


def adjust_to_side_wall(wall_threshold=0.25):
    """Try to align to a side wall (left then right) for drift correction."""
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

# -----------------------------------------------------------------------------
# 6. WALL-DETECTION HELPERS  (360 lidar)
# -----------------------------------------------------------------------------

def check_wall_ahead(threshold=0.18):
    """
    Return True if the front sector of the 360 lidar detects a wall.
    Replaces the old narrow-FOV center-ray approach.
    """
    ranges = lidar.getRangeImage()
    if not ranges:
        return False
    N = len(ranges)
    f = N / 360
    wall_map = scan_8_walls(ranges, f, N)
    # Also cross-check raw distance for tighter threshold control
    s_front = ranges[int(170*f) : int(190*f)]
    d_front = get_avg_dist(s_front)
    return wall_map['front'] == "WALL" or (not math.isinf(d_front) and d_front < threshold)

# -----------------------------------------------------------------------------
# 7. NAVIGATION
# -----------------------------------------------------------------------------

def get_shortest_path_bfs(start, target, grid_size=12):
    """BFS flood-fill on the known-wall grid."""
    from collections import deque
    if start == target:
        return [start]
    queue   = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nxt = (node[0] + dx, node[1] + dy)
            if not (0 <= nxt[0] < grid_size and 0 <= nxt[1] < grid_size):
                continue
            if get_wall_id(node, nxt) in known_walls:
                continue
            if nxt in visited:
                continue
            new_path = path + [nxt]
            if nxt == target:
                return new_path
            visited.add(nxt)
            queue.append((nxt, new_path))
    return None


def get_target_heading(current_node, next_node):
    dx = next_node[0] - current_node[0]
    dy = next_node[1] - current_node[1]
    if dy ==  1: return 0   # North
    if dx ==  1: return 1   # East
    if dy == -1: return 2   # South
    if dx == -1: return 3   # West
    return 0


def turn_to_heading(target_h):
    global current_heading
    diff = (target_h - current_heading) % 4
    if diff == 1:
        turn_left_90()
    elif diff == 2:
        turn_left_90(); turn_left_90()
    elif diff == 3:
        turn_right_90()
    current_heading = target_h


def navigate_to_target(target_x, target_y):
    global current_x, current_y, current_heading
    straight_tiles_count = 0

    while (current_x, current_y) != (target_x, target_y):
        # --- proactively register visible walls before planning ---
        scan_and_register_walls()

        path = get_shortest_path_bfs((current_x, current_y), (target_x, target_y))
        if not path or len(path) < 2:
            return False

        next_node = path[1]
        target_h  = get_target_heading((current_x, current_y), next_node)

        if target_h != current_heading:
            turn_to_heading(target_h)
            straight_tiles_count = 0

        if check_wall_ahead():
            adjust_to_wall()
            known_walls.add(get_wall_id((current_x, current_y), next_node))
            straight_tiles_count = 0
            continue

        move_forward_tiles(1)
        current_x, current_y = next_node

        # --- register walls from the new tile immediately after moving ---
        scan_and_register_walls()

        straight_tiles_count += 1
        if straight_tiles_count >= 4:
            adjust_to_side_wall()
            straight_tiles_count = 0

    print(f"Reached coordinate: ({target_x}, {target_y})")
    return True

# -----------------------------------------------------------------------------
# 8. FINAL DETECTION & MESSAGING
# -----------------------------------------------------------------------------

def detect_final_wall_color():
    cam.enable(timestep_ms)
    robot.step(timestep_ms)
    robot.step(timestep_ms)
    raw = cam.getImage()
    cam.disable()
    if not raw:
        return "Unknown"
    img = np.frombuffer(raw, np.uint8).reshape((cam.getHeight(), cam.getWidth(), 4))
    h, w = img.shape[:2]
    patch = img[h//2-10:h//2+10, w//2-10:w//2+10]
    b, g, r, _ = cv2.mean(patch)
    if max(r, g, b) < 60:  return "Black"
    if r > g and r > b:    return "Red"
    if g > r and g > b:    return "Green"
    if b > r and b > g:    return "Blue"
    return "Unknown"

# -----------------------------------------------------------------------------
# 9. GLOBAL STATE & MAIN
# -----------------------------------------------------------------------------

current_x       = 0
current_y       = 0
current_heading = 0   # 0=North 1=East 2=South 3=West
known_walls     = set()


def main():
    global current_x, current_y

    robot.step(timestep_ms)

    # Move forward until hitting the first wall, using 8-wall scan
    while not check_wall_ahead():
        move_forward_tiles(1)
        current_y += 1
        scan_and_register_walls()   # learn the surroundings at each tile

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
    print("Time taken:", robot.getTime())


if __name__ == '__main__':
    main()
