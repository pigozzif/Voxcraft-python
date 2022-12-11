import os

for dir in ["data1"]:
    for root, dirs, files in os.walk(dir):
        for file in files:
            os.system("git rm --cached {}".format(os.path.join(root, file)))