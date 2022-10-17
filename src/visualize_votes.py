import argparse
import subprocess

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

VOXEL_SIZE = 15


def draw_robot(data, width, height, touch, is_passable):
    image = Image.new("RGB", ((width + 2) * VOXEL_SIZE, (height + 2) * VOXEL_SIZE), color="white")
    draw = ImageDraw.Draw(image)
    step_count, voxels = tuple(data.split(":"))
    draw.text((0, 0), str(step_count) + "  truth", fill="black")
    draw.rectangle((((width + 2) * VOXEL_SIZE / 2 - VOXEL_SIZE / 4, 0),
                    ((width + 2) * VOXEL_SIZE / 2 + VOXEL_SIZE / 4, VOXEL_SIZE / 2)),
                   outline="white", fill="blue" if is_passable else "red")
    new_voxels = []
    for voxel in voxels.split("/")[:-1]:
        vote, x, y, z, t = tuple(voxel.split(","))
        vote = float(vote)
        x = int(x)
        y = int(y)
        z = int(z)
        y = height - y
        t = True if t == "1" else False
        new_voxels.append((vote, x, y, z, t))
    min_x, min_y = min(new_voxels, key=lambda x: x[1])[1], min(new_voxels, key=lambda x: x[2])[2]
    for vote, x, y, z, t in new_voxels:
        x = x - min_x
        y = y - min_y
        x0y0, x1y1 = (VOXEL_SIZE + x * VOXEL_SIZE, VOXEL_SIZE + y * VOXEL_SIZE), \
                     (VOXEL_SIZE + (x + 1) * VOXEL_SIZE, VOXEL_SIZE + (y + 1) * VOXEL_SIZE)
        if not touch:
            fill = "white"
        elif vote > 0.0:
            fill = "blue"
        else:
            fill = "red"
        draw.rectangle([x0y0, x1y1], outline="black" if not t else "yellow", fill=fill, width=2)
    draw.text(((width + 2) * VOXEL_SIZE * 0.66 - VOXEL_SIZE / 4, 0), "  majority", fill="black")
    majority = 1 if len(list(filter(lambda x: x[0] >= 0.0, new_voxels))) >\
                    len(list(filter(lambda x: x[0] < 0.0, new_voxels))) else 0
    draw.rectangle((((width + 2) * VOXEL_SIZE * 0.66 - VOXEL_SIZE / 4, 0),
                    ((width + 2) * VOXEL_SIZE * 0.66 + VOXEL_SIZE / 4, VOXEL_SIZE / 2)),
                   outline="white", fill="blue" if majority else "red")
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--path", default=None, type=str, help="path to the .history file")
    parser.add_argument("--width", default=9, type=int, help="width of the robot")
    parser.add_argument("--height", default=9, type=int, help="height of the robot")
    arguments = parser.parse_args()
    frame_count = 0
    images = []
    for line in open(arguments.path, "r"):
        frame_count += 1
        if ("/" not in line) or "vx3_node_worker" in line or "setting" in line or frame_count % 2 == 0:
            continue
        images.append(draw_robot(line, arguments.width, arguments.height, True, "impassable" not in arguments.path))
        print(frame_count)
    print("simulation over")
    fps = len(images) // 50
    subprocess.call("rm output.avi", shell=True)
    out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                          ((int(arguments.width) + 2) * VOXEL_SIZE, (int(arguments.height) + 2) * VOXEL_SIZE))
    for image in images:
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        out.write(cv_img)
    out.release()
