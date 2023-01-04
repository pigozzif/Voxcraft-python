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
    directory = "state"
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith("history") or not (
                    ("history0" in root and "3094" in file) or
                    ("history1" in root and "1921" in file) or
                    ("history2" in root and "3959" in file) or
                    ("history3" in root and "1880" in file)):
                continue
            points = get_trajectory(file_name=os.path.join(root, file))
            plt.plot([p[0] for p in points], [p[1] for p in points],
                     color="red" if "impassable" in file else "blue",
                     label="impassable" if "impassable" in file else "passable")
    plt.axis("off")
    plt.savefig(os.path.join(os.getcwd(), directory, "trajectories.png"))
    os.system("rm -rf frames")
