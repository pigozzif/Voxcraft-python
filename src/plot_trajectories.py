import os

import matplotlib.pyplot as plt


def get_trajectory(file_name):
    points = []
    for line in open(file_name, "r"):
        if line.startswith("P"):
            try:
                x, y = line.split(" ")[1].strip().split(",")
            except:
                continue
            points.append((float(x), float(y)))
    return points


if __name__ == "__main__":
    idx = 0
    os.system("rm -rf frames")
    os.system("mkdir frames")
    directory = "random_sensing_all"
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith("history") or not (
                    ("history0" in root and "1812" in file) or
                    ("history1" in root and "2634" in file) or
                    ("history2" in root and "3443" in file) or
                    ("history3" in root and "3976" in file)):
                continue
            points = get_trajectory(file_name=os.path.join(root, file))
            plt.plot([p[0] for p in points], [p[1] for p in points],
                     color="red" if "impassable" in file else "blue",
                     label="impassable" if "impassable" in file else "passable")
    plt.xlim(0.0, 0.25)
    plt.ylim(0.10, 0.50)
    plt.savefig(os.path.join(os.getcwd(), directory, "trajectories.png"))
    os.system("rm -rf frames")