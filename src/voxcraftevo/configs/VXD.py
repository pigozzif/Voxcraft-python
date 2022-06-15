import numpy as np
from lxml import etree


class VXD(object):

    def __init__(self, isPassable=1, NeuralWeights=None):
        root = etree.XML("<VXD></VXD>")
        self.tree = etree.ElementTree(root)
        if NeuralWeights is not None:
            self.NeuralWeights = ",".join([str(w) for w in NeuralWeights])
        else:
            self.NeuralWeights = ""
        self.isPassable = isPassable

    def set_tags(self, record_history, RecordVoxel=1, RecordLink=0, RecordFixedVoxels=1, RecordStepSize=100):
        root = self.tree.getroot()

        neural = etree.SubElement(root, "Controller")
        etree.SubElement(neural, "NeuralWeights").text = self.NeuralWeights

        task = etree.SubElement(root, "Task")
        etree.SubElement(task, "Passable").text = str(self.isPassable)

        if record_history:
            history = etree.SubElement(root, "RecordHistory")
            history.set('replace', 'VXA.Simulator.RecordHistory')
            etree.SubElement(history, "RecordStepSize").text = str(RecordStepSize)  # Capture image every 100 time steps
            etree.SubElement(history, "RecordVoxel").text = str(RecordVoxel)  # Add voxels to the visualization
            etree.SubElement(history, "RecordLink").text = str(RecordLink)  # Add links to the visualization
            etree.SubElement(history, "RecordFixedVoxels").text = str(RecordFixedVoxels)

    def set_data(self, data):
        root = self.tree.getroot()

        X_Voxels, Y_Voxels, Z_Voxels = data.shape
        body_flatten = np.zeros((X_Voxels * Y_Voxels, Z_Voxels), dtype=np.int8)
        for z in range(Z_Voxels):
            k = 0
            for y in range(Y_Voxels):
                for x in range(X_Voxels):
                    body_flatten[k, z] = data[x, y, z]
                    k += 1
        structure = etree.SubElement(root, "Structure")
        structure.set('replace', 'VXA.VXC.Structure')
        structure.set('Compression', 'ASCII_READABLE')

        etree.SubElement(structure, "X_Voxels").text = str(X_Voxels)
        etree.SubElement(structure, "Y_Voxels").text = str(Y_Voxels)
        etree.SubElement(structure, "Z_Voxels").text = str(Z_Voxels)

        # set body data
        data_tag = etree.SubElement(structure, "Data")
        for i in range(Z_Voxels):
            string = "".join([f"{c}" for c in body_flatten[:, i]])
            etree.SubElement(data_tag, "Layer").text = etree.CDATA(string)

    def write(self, filename='robot.vxd'):
        with open(filename, 'w+') as f:
            f.write(etree.tostring(self.tree, encoding="unicode", pretty_print=True))
