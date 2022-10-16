import argparse
from PIL import Image, ImageDraw, ImageFont


def draw_robot(data, width, height, touch):
    voxel_size = 20
    image = Image.new("RGB", ((width + 2) * voxel_size, (height + 2) * voxel_size), color="white")
    draw = ImageDraw.Draw(image)
    step_count, voxels = tuple(data.split(":"))
    draw.text((0, 0), str(step_count), fill="black")
    voxels = voxels.replace("}", "")
    new_voxels = []
    for voxel in voxels.split("{")[1:]:
        vote, x, y, z = tuple(voxel.split(","))
        vote = float(vote)
        x = int(x)
        y = int(y)
        z = int(z)
        y = height - y
        new_voxels.append((vote, x, y, z))
    min_x, min_y = min(new_voxels, key=lambda x: x[1])[1], min(new_voxels, key=lambda x: x[2])[2]
    for vote, x, y, z in new_voxels:
        x = x - min_x
        y = y - min_y
        x0y0, x1y1 = (voxel_size + x * voxel_size, voxel_size + y * voxel_size), \
                     (voxel_size + (x + 1) * voxel_size, voxel_size + (y + 1) * voxel_size)
        if not touch:
            fill = "white"
        elif vote > 0.0:
            fill = "blue"
        else:
            fill = "red"
        draw.rectangle([x0y0, x1y1], outline="black", fill=fill, width=2)
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
        if ("{" not in line and "}" not in line) or "setting" in line:
            continue
        print(frame_count)
        images.append(draw_robot(line, arguments.width, arguments.height, True))
    images[0].save("robot.gif", format="GIF", append_images=images, save_all=True, duration=50000 / len(images), loop=0)
