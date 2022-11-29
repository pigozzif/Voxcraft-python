import numpy as np
from lxml import etree


class VXD(object):

    def __init__(self, terrainID: int, isPassable: int = 1, NeuralWeights: np.ndarray = None, age: int = 0):
        root = etree.XML("<VXD></VXD>")
        self.tree = etree.ElementTree(root)
        if NeuralWeights is not None:
            self.NeuralWeights = ",".join([str(w) for w in NeuralWeights])
        else:
            self.NeuralWeights = ""
        self.isPassable = isPassable
        self.terrainID = terrainID
        self.age = age

    def set_tags(self, RecordVoxel: int = 1, RecordLink: int = 0, RecordFixedVoxels: int = 1,
                 RecordStepSize: float = 100) -> None:
        root = self.tree.getroot()

        neural = etree.SubElement(root, "Controller")
        etree.SubElement(neural, "NeuralWeights").text = self.NeuralWeights
        etree.SubElement(neural, "Age").text = self.age

        task = etree.SubElement(root, "Task")
        etree.SubElement(task, "Passable").text = "1" if self.isPassable else "0"
        etree.SubElement(task, "TerrainID").text = str(self.terrainID)

        history = etree.SubElement(root, "RecordHistory")
        history.set('replace', 'VXA.Simulator.RecordHistory')
        etree.SubElement(history, "RecordStepSize").text = str(RecordStepSize)  # Capture image every 100 time steps
        etree.SubElement(history, "RecordVoxel").text = str(RecordVoxel)  # Add voxels to the visualization
        etree.SubElement(history, "RecordLink").text = str(RecordLink)  # Add links to the visualization
        etree.SubElement(history, "RecordFixedVoxels").text = str(RecordFixedVoxels)

    def set_data(self, data: np.ndarray) -> None:
        root = self.tree.getroot()

        x_voxels, y_voxels, z_voxels = data.shape
        body_flatten = np.zeros((x_voxels * y_voxels, z_voxels), dtype=np.int8)
        for z in range(z_voxels):
            k = 0
            for y in range(y_voxels):
                for x in range(x_voxels):
                    body_flatten[k, z] = data[x, y, z]
                    k += 1
        structure = etree.SubElement(root, "Structure")
        structure.set('replace', 'VXA.VXC.Structure')
        structure.set('Compression', 'ASCII_READABLE')

        etree.SubElement(structure, "X_Voxels").text = str(x_voxels)
        etree.SubElement(structure, "Y_Voxels").text = str(y_voxels)
        etree.SubElement(structure, "Z_Voxels").text = str(z_voxels)

        # set body data
        data_tag = etree.SubElement(structure, "Data")
        for i in range(z_voxels):
            string = "".join([f"{c}" for c in body_flatten[:, i]])
            etree.SubElement(data_tag, "Layer").text = etree.CDATA(string)

    def write(self, filename='robot.vxd'):
        with open(filename, 'w+') as f:
            f.write(etree.tostring(self.tree, encoding="unicode", pretty_print=True))
