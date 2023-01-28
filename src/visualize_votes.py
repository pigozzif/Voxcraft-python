import os
import sys

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
    majority = 1 if len(list(filter(lambda x: x[0] >= 0.0, new_voxels))) > len(
        list(filter(lambda x: x[0] < 0.0, new_voxels))) else 0
    draw.rectangle((((width + 2) * VOXEL_SIZE * 0.66 - VOXEL_SIZE / 4, 0),
                    ((width + 2) * VOXEL_SIZE * 0.66 + VOXEL_SIZE / 4, VOXEL_SIZE / 2)),
                   outline="white", fill="blue" if majority else "red")
    return image


def create_video(path, width, height):
    frame_count = 0
    images = []
    for line in open(path, "r"):
        frame_count += 1
        if ("/" not in line) or "vx3_node_worker" in line or "setting" in line or frame_count % 5 == 0 \
                or line.startswith("?"):
            continue
        try:
            im = draw_robot(line, width, height, True, "-0" not in path)
            images.append(im)
        except:
            print("FAULTY FRAME: {}".format(frame_count))
            raise
    fps = len(images) // 50
    if fps < 1:
        print("Not enough votes for {}".format(path))
        return
    out = cv2.VideoWriter("{}.avi".format(path.split("/")[-1].split(".")[0]),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                          ((int(width) + 2) * VOXEL_SIZE, (int(height) + 2) * VOXEL_SIZE))
    for image in images:
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        out.write(cv_img)
    out.release()


def get_body_length(shape):
    if shape == "flatworm":
        return 4
    elif shape == "starfish":
        return 9
    elif shape == "gecko":
        return 6
    raise ValueError("Unknown shape: {}".format(shape))


def get_best_sensing_id(directory):
    bests = {}
    for file in os.listdir(directory):
        if file.endswith("csv"):
            last_line = open(os.path.join(directory, file), "r").readlines()[-1]
            bests[last_line.split(";")[0]] = last_line.split(";")[3]
    return bests


if __name__ == "__main__":
    directory = sys.argv[1]
    bests = get_best_sensing_id(directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "passable_right" in file or not any([("history" + s in root and b in file) for s, b in bests.items()]):
                continue
            body_length = get_body_length(file.split("-")[2])
            create_video(os.path.join(os.getcwd(), root, file), body_length, body_length)
