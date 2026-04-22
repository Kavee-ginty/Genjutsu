from controller import Robot

# 1. Initialize Robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# 2. Setup Emitter (Required for sending messages to supervisor)
# Make sure the device name matches the emitter name on your Webots node
emitter = robot.getDevice('emitter_2')

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

print("E-puck: Controller initialized. Waiting for contestant logic...")

# 3. Main Loop
while robot.step(timestep) != -1:
    
    # ---------------------------------------------------------
    # CONTESTANTS: Add your sensor reading, movement, and logic here.
    # ---------------------------------------------------------
    
    # Example of sending a message when your custom condition is met:
    # if my_wall_detected_condition:
    #     send_message("Red")

    pass