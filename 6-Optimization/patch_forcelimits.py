import re

path = r"X:\Xiyuan_Wang\1-LIBR\9-Centipede_project\3-Centipede_main\2-Simulation\Centipede_MUJOCO-main\Centipede_MUJOCO-main\3-Model\2-Centipede_FARMS\centipede.xml"
with open(path, 'r') as f:
    content = f.read()

# Replace stiffness on all pitch joints
content = re.sub(
    r'(name="joint_(?:pitch_body|passive)_\d+"[^/]*?)stiffness="[^"]*"',
    r'\1stiffness="1e-2"',
    content
)

with open(path, 'w') as f:
    f.write(content)
print("Done")