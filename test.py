import numpy as np

angles = [5.001* np.pi]

for angle in angles:
    angle = angle % (2 * np.pi) / np.pi
    if angle > 1:
        angle -= 2
    print(angle)
