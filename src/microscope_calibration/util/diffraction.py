import numpy as np
from CifFile import ReadCif
from diffpy.structure import loadStructure
from diffsims.generators.simulation_generator import SimulationGenerator
from diffsims.generators.zap_map_generator import get_rotation_from_z_to_direction
from orix.crystal_map import Phase
from orix.quaternion import Rotation


# See also
# https://github.com/py4dstem/py4DSTEM_tutorials/blob/main/notebooks/basics_03_calibration.ipynb
def get_twothetas(cif_filename, acceleration_voltage_V, reciprocal_radius=1):
    gen = SimulationGenerator(
        accelerating_voltage=acceleration_voltage_V / 1000,
        precession_angle=10,
        minimum_intensity=0.0001,
    )
    structure_raw = ReadCif(cif_filename)
    key = list(structure_raw.keys())[0]
    space_group = int(structure_raw[key]["_space_group_IT_number"])
    structure = loadStructure(cif_filename)
    p = Phase(structure=structure, space_group=space_group)
    thetas = set()
    for ha in (0, 1, 2, 3, 4, 5):
        for ka in (0, 1, 2, 3):
            for el in (0, 1, 2):
                euler = get_rotation_from_z_to_direction(p.structure, [ha, ka, el])
                rot = Rotation.from_euler(euler, degrees=True)
                sim = gen.calculate_diffraction2d(
                    phase=p,
                    rotation=rot,
                    reciprocal_radius=reciprocal_radius,
                    # This seems to avoid errorneous reflections while still including enough peaks
                    max_excitation_error=0.0005,
                )
                sim.coordinates.calculate_theta(voltage=acceleration_voltage_V)
                thetas_with_intensity = [
                    item[1]
                    for item in zip(sim.coordinates.intensity, sim.coordinates.theta)
                    if item[0] > 1
                ]
                thetas.update(np.round(thetas_with_intensity, decimals=5))
    # diffsims calculates theta, not twotheta
    return np.array(sorted(thetas)) * 2
