import os

for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".avi") or file.endswith(".pickle"):
            os.system("git rm --cached {}".format(os.path.join(root, file)))
