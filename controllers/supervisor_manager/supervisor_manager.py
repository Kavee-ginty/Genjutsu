from controller import Supervisor
import random

# 1. Initialize Supervisor
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# 2. Setup Receiver
receiver = robot.getDevice('receiver')
receiver.enable(timestep)

# 3. Setup the Treasure Node
treasure_node = robot.getFromDef("Treasure")
treasure_translation = None
treasure_transparency_field = None

if treasure_node:
    treasure_translation = treasure_node.getField("translation")

    # Get Transform node
    treasure_children = treasure_node.getField("children")
    transform_node = treasure_children.getMFNode(0)

    # Get Shape node
    transform_children = transform_node.getField("children")
    shape_node = transform_children.getMFNode(0)

    # Get Appearance node
    appearance_node = shape_node.getField("appearance").getSFNode()

    # Get transparency field
    treasure_transparency_field = appearance_node.getField("transparency")

    # Hide treasure initially
    if treasure_translation:
        treasure_translation.setSFVec3f([0, 0, 1000.0])

    if treasure_transparency_field:
        treasure_transparency_field.setSFFloat(1.0)  # fully invisible

else:
    print("Supervisor Error: Could not find DEF 'Treasure'.")

# 4. Define Config and Wall Setup
config = {
    "Black": [0.0, 0.0, 0.0],
    "Blue":  [0.0, 0.0, 1.0],
    "Red":   [1.0, 0.0, 0.0],
    "Green": [0.0, 1.0, 0.0]
}

wall = robot.getFromDef("My_Wall")
target_name = ""

if wall:
    appearance = wall.getField("appearance").getSFNode()
    color_field = appearance.getField("baseColor")

    target_name = random.choice(list(config.keys()))
    color_field.setSFColor(config[target_name])

    print("--- SIMULATION STARTED ---")
    print(f"Supervisor: The target is {target_name}. Waiting for E-puck...")

# 5. Main Loop
while robot.step(timestep) != -1:

    if receiver.getQueueLength() > 0:
        received_color = receiver.getString()

        if received_color == target_name:
            print(f"Supervisor: Match! E-puck found {received_color}. Showing treasure!")

            # Teleport treasure
            if treasure_translation:
                treasure_translation.setSFVec3f([-1.13, 0.92, 0.0])

            # Make treasure visible
            if treasure_transparency_field:
                treasure_transparency_field.setSFFloat(0.0)

        receiver.nextPacket()
