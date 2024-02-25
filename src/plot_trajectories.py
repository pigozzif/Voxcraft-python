import os
import sys

import matplotlib.pyplot as plt


def get_trajectory(file_name):
    points = []
    for line in open(file_name, "r"):
        if line.startswith("P"):
            try:
                x, y = line.split(" ")[1].strip().split(",")
                points.append((float(x), float(y)))
            except:
                continue
    return points


def get_best_sensing_id(directory):
    bests = {}
    for file in os.listdir(directory):
        if file.endswith("csv"):
            last_line = open(os.path.join(directory, file), "r").readlines()[-1]
            bests[last_line.split(";")[0]] = last_line.split(";")[4]
    return bests


if __name__ == "__main__":
    os.system("rm -rf frames")
    os.system("mkdir frames")
    directory = sys.argv[1]
    bests = get_best_sensing_id(directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith("history") or \
                    not any([("history" + s in root and b in file) for s, b in bests.items()]):
                continue
            points = get_trajectory(file_name=os.path.join(root, file))
            if "impassable" in file:
                color = "red"
            elif "right" in file:
                color = "blue"
            else:
                color = "green"
            plt.plot([p[0] for p in points], [p[1] for p in points],
                     linewidth=1.5,
                     color=color,
                     label="impassable" if "impassable" in file else "passable")
    plt.xlim(0.09, 0.175)
    plt.ylim(0.15, 0.35)
    plt.axis("off")
    plt.savefig(os.path.join(os.getcwd(), directory, "trajectories.png"))
    os.system("rm -rf frames")
