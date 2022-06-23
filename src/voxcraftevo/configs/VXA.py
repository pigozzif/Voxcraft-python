import numpy as np
from lxml import etree

'''
Does not yet include signaling parameters
'''


class VXA(object):

    def __init__(self, HeapSize: float = 0.1, EnableCilia: int = 0, EnableExpansion: int = 1, DtFrac: float = 0.95,
                 BondDampingZ: float = 1, ColDampingZ: float = 0.8, SlowDampingZ: float = 0.01,
                 EnableCollision: int = 0, SimTime: int = 5, TempPeriod: float = 0.1, GravEnabled: int = 1,
                 GravAcc: float = -9.81, FloorEnabled: int = 1, Lattice_Dim: float = 0.01, RecordStepSize: int = 100,
                 RecordVoxel: int = 1, RecordLink: int = 0, RecordFixedVoxels: int = 1, VaryTempEnabled: int = 1,
                 TempAmplitude: float = 20, TempBase: float = 25, TempEnabled: int = 1):

        root = etree.XML("<VXA></VXA>")
        root.set('Version', '1.1')
        self.tree = etree.ElementTree(root)

        self.HeapSize = HeapSize
        self.EnableCilia = EnableCilia
        self.EnableExpansion = EnableExpansion
        self.DtFrac = DtFrac
        self.BondDampingZ = BondDampingZ
        self.ColDampingZ = ColDampingZ
        self.SlowDampingZ = SlowDampingZ
        self.EnableCollision = EnableCollision
        self.SimTime = SimTime
        self.TempPeriod = TempPeriod
        self.GravEnabled = GravEnabled
        self.GravAcc = GravAcc
        self.FloorEnabled = FloorEnabled
        self.Lattice_Dim = Lattice_Dim
        self.RecordStepSize = RecordStepSize
        self.RecordVoxel = RecordVoxel
        self.RecordLink = RecordLink
        self.RecordFixedVoxels = RecordFixedVoxels
        self.VaryTempEnabled = VaryTempEnabled
        self.TempAmplitude = TempAmplitude
        self.TempBase = TempBase
        self.TempEnabled = TempEnabled

        self.NextMaterialID = 1  # Material ID's start at 1, 0 denotes empty space

        self.set_default_tags()

    def set_default_tags(self) -> None:
        root = self.tree.getroot()

        # GPU
        gpu = etree.SubElement(root, 'GPU')
        etree.SubElement(gpu, "HeapSize").text = str(self.HeapSize)

        # Simulator
        simulator = etree.SubElement(root, "Simulator")
        etree.SubElement(simulator, "EnableCilia").text = str(self.EnableCilia)
        etree.SubElement(simulator, "EnableExpansion").text = str(
            self.EnableExpansion)  # 0 only contraction, 1 is contration + expansion

        integration = etree.SubElement(simulator, "Integration")
        etree.SubElement(integration, "DtFrac").text = str(self.DtFrac)

        damping = etree.SubElement(simulator, "Damping")
        etree.SubElement(damping, "BondDampingZ").text = str(self.BondDampingZ)
        etree.SubElement(damping, "ColDampingZ").text = str(self.ColDampingZ)
        etree.SubElement(damping, "SlowDampingZ").text = str(self.SlowDampingZ)

        attach_detach = etree.SubElement(simulator, "AttachDetach")
        etree.SubElement(attach_detach, "EnableCollision").text = str(self.EnableCollision)

        stop_condition = etree.SubElement(simulator, "StopCondition")
        formula = etree.SubElement(stop_condition, "StopConditionFormula")
        sub = etree.SubElement(formula, "mtSUB")
        etree.SubElement(sub, "mtVAR").text = 't'
        etree.SubElement(sub, "mtCONST").text = str(self.SimTime)

        fitness = etree.SubElement(simulator, "FitnessFunction")  # default - maximum x distance
        add = etree.SubElement(fitness, "mtADD")
        mul = etree.SubElement(add, 'mtMUL')
        etree.SubElement(mul, "mtVAR").text = 'x'
        etree.SubElement(mul, "mtVAR").text = 'x'
        mul2 = etree.SubElement(add, 'mtMUL')
        etree.SubElement(mul2, "mtVAR").text = 'y'
        etree.SubElement(mul2, "mtVAR").text = 'y'

        history = etree.SubElement(simulator, "RecordHistory")
        etree.SubElement(history, "RecordStepSize").text = str(
            self.RecordStepSize)  # Capture image every 100 time steps
        etree.SubElement(history, "RecordVoxel").text = str(self.RecordVoxel)  # Add voxels to the visualization
        etree.SubElement(history, "RecordLink").text = str(self.RecordLink)  # Add links to the visualization
        etree.SubElement(history, "RecordFixedVoxels").text = str(self.RecordFixedVoxels)

        # Environment

        environment = etree.SubElement(root, "Environment")
        thermal = etree.SubElement(environment, "Thermal")
        etree.SubElement(thermal, "TempEnabled").text = str(self.TempEnabled)
        etree.SubElement(thermal, "VaryTempEnabled").text = str(self.VaryTempEnabled)
        etree.SubElement(thermal, "TempPeriod").text = str(self.TempPeriod)
        etree.SubElement(thermal, "TempAmplitude").text = str(self.TempAmplitude)
        etree.SubElement(thermal, "TempBase").text = str(self.TempBase)

        gravity = etree.SubElement(environment, "Gravity")
        etree.SubElement(gravity, "GravEnabled").text = str(self.GravEnabled)
        etree.SubElement(gravity, "GravAcc").text = str(self.GravAcc)
        etree.SubElement(gravity, "FloorEnabled").text = str(self.FloorEnabled)

        # VXC tags
        vxc = etree.SubElement(root, "VXC")
        vxc.set("Version", "0.94")

        lattice = etree.SubElement(vxc, "Lattice")
        etree.SubElement(lattice, "Lattice_Dim").text = str(self.Lattice_Dim)

        # Materials
        etree.SubElement(vxc, "Palette")

        # Structure
        structure = etree.SubElement(vxc, "Structure")
        structure.set("Compression", "ASCII_READABLE")
        # set some default data
        etree.SubElement(structure, "X_Voxels").text = "1"
        etree.SubElement(structure, "Y_Voxels").text = "1"
        etree.SubElement(structure, "Z_Voxels").text = "2"

        data = etree.SubElement(structure, "Data")
        etree.SubElement(data, "Layer").text = etree.CDATA("0")
        etree.SubElement(data, "Layer").text = etree.CDATA("1")

    def add_material(self, E: float = 10000, RHO: float = 1000, P: float = 0.35, CTE: float = 0, uStatic: float = 1,
                     uDynamic: float = 0.8, isSticky: int = 0, hasCilia: int = 0, isBreakable: int = 0,
                     isMeasured: int = 1, RGBA: tuple = None, isFixed: int = 0, TempPhase: float = 0) -> int:

        material_id = self.NextMaterialID
        self.NextMaterialID += 1

        if RGBA is None:
            # assign the material a random color
            RGBA = np.around((np.random.random(), np.random.random(), np.random.random(), 1), 2)
        else:
            if len(RGBA) == 3:  # if no alpha, add alpha of 255
                RGBA = (RGBA[0], RGBA[1], RGBA[2], 255)

            # normalize between 0-1
            RGBA = (RGBA[0] / 255, RGBA[1] / 255, RGBA[2] / 255, RGBA[3] / 255)

        palette = self.tree.find("*/Palette")
        material = etree.SubElement(palette, "Material")

        etree.SubElement(material, "Name").text = str(material_id)

        display = etree.SubElement(material, "Display")
        etree.SubElement(display, "Red").text = str(RGBA[0])
        etree.SubElement(display, "Green").text = str(RGBA[1])
        etree.SubElement(display, "Blue").text = str(RGBA[2])
        etree.SubElement(display, "Alpha").text = str(RGBA[3])

        mechanical = etree.SubElement(material, "Mechanical")
        etree.SubElement(mechanical, "isMeasured").text = str(
            isMeasured)  # if material should be included in fitness function
        etree.SubElement(mechanical, "Fixed").text = str(isFixed)
        etree.SubElement(mechanical, "sticky").text = str(isSticky)
        etree.SubElement(mechanical, "Cilia").text = str(hasCilia)
        etree.SubElement(mechanical, "MatModel").text = str(isBreakable)  # 0 = no failing
        etree.SubElement(mechanical, "Elastic_Mod").text = str(E)
        etree.SubElement(mechanical, "Fail_Stress").text = "0"  # no fail if matModel is 0
        etree.SubElement(mechanical, "Density").text = str(RHO)
        etree.SubElement(mechanical, "Poissons_Ratio").text = str(P)
        etree.SubElement(mechanical, "CTE").text = str(CTE)
        etree.SubElement(mechanical, "MaterialTempPhase").text = str(TempPhase)
        etree.SubElement(mechanical, "uStatic").text = str(uStatic)
        etree.SubElement(mechanical, "uDynamic").text = str(uDynamic)

        return material_id

    def write(self, filename: str = 'base.vxa') -> None:

        # If no material has been added, add default material
        if self.NextMaterialID == 0:
            self.add_material()

        with open(filename, 'w+') as f:
            f.write(etree.tostring(self.tree, encoding="unicode", pretty_print=True))
