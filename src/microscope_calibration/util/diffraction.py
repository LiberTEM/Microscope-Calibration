import numpy as np

from CifFile import ReadCif
from diffpy.structure import loadStructure
from orix.crystal_map import Phase
from orix.quaternion import Rotation
from diffsims.generators.simulation_generator import SimulationGenerator


# See also
# https://github.com/py4dstem/py4DSTEM_tutorials/blob/main/notebooks/basics_03_calibration.ipynb
def get_twothetas(cif_filename, acceleration_voltage_V, reciprocal_radius=1):
    gen = SimulationGenerator(
        accelerating_voltage=acceleration_voltage_V / 1000,
    )
    structure_raw = ReadCif(cif_filename)
    key = list(structure_raw.keys())[0]
    space_group = int(structure_raw[key]['_space_group_IT_number'])
    structure = loadStructure('EntryWithCollCode163723.cif')
    p = Phase(structure=structure, space_group=space_group)
    twothetas = set()
    rng = np.random.default_rng(seed=0)
    eulers = rng.uniform(0.0, 2 * np.pi, (10, 3))
    for euler in eulers:
        rot = Rotation.from_euler(euler)
        sim = gen.calculate_diffraction2d(
            phase=p, rotation=rot,
            reciprocal_radius=reciprocal_radius,
            # Large excitation error to capture many peaks
            max_excitation_error=1,
        )
        sim.coordinates.calculate_theta(voltage=acceleration_voltage_V)
        twothetas.update(np.round(sim.coordinates.theta, decimals=5))
    return np.array(sorted(twothetas))
