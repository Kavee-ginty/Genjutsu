# Improved version of the e‑puck controller.
#
# The goal of this version is to reduce the time spent in tight polling loops
# and sensor processing without increasing the maximum wheel speed.  Instead of
# repeatedly querying the GPS at every control step to decide when to stop,
# this controller computes in advance how many control steps are needed for a
# given travel distance and only reads the GPS when necessary.  It also
# disables CPU‑intensive sensors (like the camera and LiDAR) when not in use
# and allows the caller to adjust sensor update periods, which Webots
# documentation notes can speed up simulation【280235185990496†L199-L210】.  The
# overall path remains the same as the original, but the robot should reach
# its destination sooner because less time is spent in controller overhead.

import math
import cv2
import numpy as np
from controller import Robot

# -----------------------------------------------------------------------------
# 1. INITIALIZATION & CONSTANTS
# -----------------------------------------------------------------------------

# Create the robot instance and determine the simulation time step.  Webots
# returns the basic time step in milliseconds; we convert it to seconds for
# clarity.
robot = Robot()
timestep_ms = int(robot.getBasicTimeStep())
timestep = timestep_ms / 1000.0

tile_length = 0.25  # distance between maze tiles in metres
wheel_radius = 0.0205  # radius of the e‑puck's wheels (m)

# Sensors and actuators
emitter = robot.getDevice('emitter_2')
cam = robot.getDevice('camera')
gps = robot.getDevice('gps')
compass = robot.getDevice('compass')
lidar = robot.getDevice('lidar')
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

# Optional position sensors for motors.  These sensors allow us to run
# movements in position‑control mode: we command a target rotation and
# monitor the motor positions to know when the motion is complete.  If you
# decide not to use position control you can ignore these sensors, but
# enabling them here makes them available for the functions below.
left_ps = left_motor.getPositionSensor()
right_ps = right_motor.getPositionSensor()
left_ps.enable(timestep_ms)
right_ps.enable(timestep_ms)

# Enable only what is necessary.  By default we disable the camera and LiDAR
# until we need them.  According to the Webots controller documentation, you
# can disable a device at any time to reduce computational load and speed up
# simulation【280235185990496†L199-L210】.  You can re‑enable them later with a
# larger update period when you need fresh data.
gps.enable(timestep_ms)
compass.enable(timestep_ms)
lidar.enablePointCloud()
lidar.disable()  # keep LiDAR disabled until we need to adjust to a wall
cam.disable()    # disable the camera for now

# Perform an initial step to update sensor values before using them.  This
# ensures that the compass has a valid reading so we can record the
# initial heading for later correction.  We do this only once at startup.
robot.step(timestep_ms)
initial_heading = 0.0
if compass:
    initial_vals = compass.getValues()
    initial_heading = math.atan2(initial_vals[0], initial_vals[1])

# Define a starting GPS coordinate for the maze.  These values should be
# customised to match the coordinate of the (0,0) tile in your world.
start_gps_x = 1.380
start_gps_y = -1.388

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Precompute the number of control steps needed to travel exactly one tile at a
# given wheel speed.  The linear speed of the robot is the wheel angular
# velocity (rad/s) times the wheel radius.  Dividing the tile length by the
# linear speed gives us the travel time.  Multiplying by the control step
# frequency tells us how many control loops are required.  Rounding to the
# nearest integer yields the number of wb_robot_step calls to perform.
def steps_for_distance(distance: float, angular_speed: float) -> int:
    linear_speed = abs(angular_speed) * wheel_radius
    if linear_speed <= 0.0:
        return 0
    travel_time = distance / linear_speed
    return max(1, int(round(travel_time / timestep)))


# -----------------------------------------------------------------------------
# 2. ARUCO TAG DETECTION (OPTIONAL)
# -----------------------------------------------------------------------------

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

def scan_aruco_tag(wall_threshold: float = 0.3, scan_distance: float = 0.06) -> tuple:
    """
    Attempt to detect an ArUco tag on any wall adjacent to the robot.  The
    robot uses its LiDAR to determine whether there is an obstacle close
    enough to be considered a wall (less than ``wall_threshold`` metres
    away).  If a wall is present, it will back up by ``scan_distance`` metres,
    take a snapshot with the camera and attempt to decode an ArUco tag, then
    move forward the same distance to restore its original position.  
    
    If no wall is present, OR if a wall is present but has no tag, the robot 
    rotates 90° clockwise and repeats the test. After trying all four directions, 
    the function returns ``(None, None, None, None)``. When a tag is detected 
    the function returns ``(x, y, tag_id, binary_string)``.
    """
    def detect_once() -> tuple:
        """Capture a single camera frame and detect ArUco markers."""
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

    def move_distance(distance: float, speed: float = 2.0) -> None:
        """Move the robot forward or backward by a specified amount."""
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

    lidar.enable(timestep_ms)
    robot.step(timestep_ms)
    turns = 0  
    
    # Try up to four directions: front, right, back, and left.
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
                
                # CHECK IF A TAG WAS ACTUALLY FOUND
                if result[2] is not None: 
                    # Tag found! Restore orientation and return.
                    for _ in range(turns):
                        turn_left_90(correct=False)
                    lidar.disable()
                    return result
                
                # If no tag was found, it simply skips this block and continues
                # to the turn_right_90() call below to check the next wall.

        # No wall detected OR wall had no tag: rotate 90° and try the next side.
        turn_right_90(correct=False)
        turns += 1
        
    # No tag found on ANY side. Restore orientation and disable LiDAR.
    for _ in range(turns):
        turn_left_90(correct=False)
    lidar.disable()
    
    return None, None, None, None

# -----------------------------------------------------------------------------
# 3. MOVEMENT PRIMITIVES
# -----------------------------------------------------------------------------

def move_forward_tiles(num_tiles: int, speed: float = 3.0) -> None:
    """
    Move forward a given number of tiles without continuously polling the GPS.
    This routine computes how many control steps correspond to one tile at the
    given wheel speed and runs the motors for that duration.  Because it no
    longer checks the GPS at every step, it reduces overhead and allows the
    robot to spend more of its time actually moving.  Note that the wheels
    still run at the same angular velocity as before, so the robot's physical
    speed is unchanged.
    """
    steps_per_tile = steps_for_distance(tile_length, speed)
    # iterate tile by tile
    for _ in range(num_tiles):
        left_motor.setVelocity(speed)
        right_motor.setVelocity(speed)
        for _ in range(steps_per_tile):
            if robot.step(timestep_ms) == -1:
                return
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        # allow one extra step to update the sensors
        robot.step(timestep_ms)


def move_forward_tiles_position(num_tiles: int, speed: float = 3.0) -> None:
    """
    Move forward a given number of tiles using the motor's built‑in position
    controller instead of manually counting control steps.  For each tile the
    required wheel rotation is computed as `tile_length / wheel_radius` radians.
    The current motor positions are read from the position sensors and the
    target positions are commanded accordingly.  The motors are run at the
    specified angular velocity until both wheels have reached their targets.
    Because Webots handles the low‑level servoing, no GPS polling is needed.
    """
    # rotation required to travel exactly one tile
    rotation = tile_length / wheel_radius
    for _ in range(num_tiles):
        # compute the absolute target positions for each motor
        current_left = left_ps.getValue()
        current_right = right_ps.getValue()
        target_left = current_left + rotation
        target_right = current_right + rotation
        # set the velocity and desired position; motors will begin moving
        left_motor.setVelocity(speed)
        right_motor.setVelocity(speed)
        left_motor.setPosition(target_left)
        right_motor.setPosition(target_right)
        while True:
            # step the simulation and check if the motion has completed
            if robot.step(timestep_ms) == -1:
                return
            # When the difference between current and target positions is small, stop
            if abs(left_ps.getValue() - target_left) < 0.02 and abs(right_ps.getValue() - target_right) < 0.02:
                break
        # stop motors after each tile
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        robot.step(timestep_ms)
        # restore velocity control before the next command
        left_motor.setPosition(float('inf'))
        right_motor.setPosition(float('inf'))


def move_backward_tiles(num_tiles: int, speed: float = 3.0) -> None:
    """
    Move backward a given number of tiles using time‑based control.  The
    absolute value of ``speed`` is used to compute the number of control steps
    required per tile, but the motors are driven with a negative velocity to
    move backwards.  This avoids polling the GPS at every step.
    """
    steps_per_tile = steps_for_distance(tile_length, speed)
    for _ in range(num_tiles):
        left_motor.setVelocity(-abs(speed))
        right_motor.setVelocity(-abs(speed))
        for _ in range(steps_per_tile):
            if robot.step(timestep_ms) == -1:
                return
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        robot.step(timestep_ms)


def move_backward_tiles_position(num_tiles: int, speed: float = 3.0) -> None:
    """
    Move backward a given number of tiles using the motor's position controller.
    Each wheel rotates backwards by ``tile_length / wheel_radius`` radians per
    tile.  The motors are driven at the given angular speed while Webots
    internally regulates the position.
    """
    rotation = tile_length / wheel_radius
    for _ in range(num_tiles):
        current_left = left_ps.getValue()
        current_right = right_ps.getValue()
        target_left = current_left - rotation
        target_right = current_right - rotation
        left_motor.setVelocity(abs(speed))
        right_motor.setVelocity(abs(speed))
        left_motor.setPosition(target_left)
        right_motor.setPosition(target_right)
        while True:
            if robot.step(timestep_ms) == -1:
                return
            if abs(left_ps.getValue() - target_left) < 0.02 and abs(right_ps.getValue() - target_right) < 0.02:
                break
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        robot.step(timestep_ms)
        # restore velocity control before the next command
        left_motor.setPosition(float('inf'))
        right_motor.setPosition(float('inf'))


def getXYfromgps() -> list:
    """
    Compute the integer (x, y) tile coordinates on a 12×12 grid based on the
    current GPS reading.  The maze origin (0,0) corresponds to the
    ``start_gps_x`` and ``start_gps_y`` variables defined at initialisation.
    Distances are quantised to tile_length and clipped to the range [0,11].
    """
    gps_vals = gps.getValues()
    rel_x = start_gps_x - gps_vals[0]
    rel_y = gps_vals[1] - start_gps_y
    new_x = int(round(rel_x / tile_length))
    new_y = int(round(rel_y / tile_length))
    # clip to grid bounds
    new_x = max(0, min(new_x, 11))
    new_y = max(0, min(new_y, 11))
    return [new_x, new_y]


def adjust_to_wall(target_distance: float = 0.128, tolerance: float = 0.02,
                   check_wall: bool = True, wall_threshold: float = 0.3) -> None:
    """
    Adjust the robot's position so that the distance to the wall in front is
    about ``target_distance`` metres.  If ``check_wall`` is True the LiDAR is
    enabled and a single range image is scanned first; if the object in front
    is farther than ``wall_threshold`` metres or the reading is infinite the
    adjustment is skipped.  Otherwise, a simple proportional controller is used
    to move forward/backward until the error falls within ``tolerance``.  The
    LiDAR is disabled when finished.
    """
    # Enable LiDAR and fetch an initial scan
    lidar.enable(timestep_ms)
    robot.step(timestep_ms)
    # Optional check for a wall before adjusting
    if check_wall:
        ranges = lidar.getRangeImage()
        if ranges:
            center = len(ranges) // 2
            front = ranges[center]
            # If no wall or too far, skip adjustment
            if math.isinf(front) or front > wall_threshold:
                lidar.disable()
                return
    max_speed = 1.5  # allow higher correction speed to finish sooner
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
        # proportional controller for adjustment
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


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def turn_left_90(correct: bool = True, tolerance: float = 0.05) -> None:
    """
    Rotate the robot 90 degrees counterclockwise.  After the initial turn the
    compass reading is compared against the closest multiple of 90° from the
    ``initial_heading``.  If the deviation exceeds ``tolerance`` radians and
    ``correct`` is True, a slower corrective rotation is performed to reduce the
    offset.  This mirrors the post‑turn correction logic in the original
    controller while still allowing a larger tolerance on the main rotation.
    """
    # Determine current and target headings
    vals = compass.getValues()
    current_heading = math.atan2(vals[0], vals[1])
    target_heading = normalize_angle(current_heading + math.pi / 2)
    # Perform primary rotation
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
    # Post‑turn correction relative to initial heading
    if correct:
        robot.step(timestep_ms)
        vals = compass.getValues()
        current_heading = math.atan2(vals[0], vals[1])
        # Angle relative to initial heading
        rel_angle = normalize_angle(current_heading - initial_heading)
        # Closest multiple of 90° (0, ±pi/2, ±pi)
        multiples = [0.0, math.pi/2, math.pi, -math.pi/2, -math.pi]
        closest = min(multiples, key=lambda x: abs(normalize_angle(rel_angle - x)))
        offset = normalize_angle(rel_angle - closest)
        # If offset too large, apply correction
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
    """
    Rotate the robot 90 degrees clockwise.  After the primary rotation the
    heading is compared with the nearest multiple of 90° from the initial
    heading.  If the offset exceeds ``tolerance`` and ``correct`` is True, a
    slower corrective rotation is applied.
    """
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


# -----------------------------------------------------------------------------
# 4. MAIN EXECUTION
# -----------------------------------------------------------------------------

def main() -> None:
    """
    A simple demonstration sequence exercising several of the available
    primitives.  The robot moves forward, scans its GPS position, optionally
    adjusts to a wall, turns and moves backward.  You can modify this logic
    depending on your maze and objectives.
    """
    while robot.step(timestep_ms) != -1:

        
        print(scan_aruco_tag())  # optional tag scanning
        
        break


if __name__ == '__main__':
    main()