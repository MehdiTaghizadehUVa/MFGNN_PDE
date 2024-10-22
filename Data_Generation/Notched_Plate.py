import numpy as np
import pickle
from ansys.mapdl.core import launch_mapdl
import time

# Set the seed number
seed_number = 23
np.random.seed(seed_number)

def generate_training_data(force_range: tuple, ratio_range: tuple, num_samples: int):
    """
    Generates training data for notched plates using ANSYS MAPDL.
    
    Args:
        force_range (tuple): Range of forces to be applied (min, max).
        ratio_range (tuple): Range of notch diameter to width ratios (min, max).
        num_samples (int): Number of training samples to generate.
    """
    file_name_template = 'MF_training_data_{}.pkl'  # Template for file names
    mesh_refinements = {'HF': (60, 40), 'LF': (15, 10)}  # Fine and coarse mesh sizes

    for i in range(num_samples):
        # Generate random parameters within the specified ranges
        length = 0.4
        width = 0.1
        force = np.random.uniform(*force_range)
        ratio = np.random.uniform(*ratio_range)
        notch_diameter = width * ratio  # Diameter of the notch
        notch_radius = notch_diameter / 2

        for refinement, (hole_ratio, plate_ratio) in mesh_refinements.items():
            file_name = file_name_template.format(refinement)  # File name for current refinement
            mapdl = launch_mapdl()

            # Element Type and Material Properties
            mapdl.prep7()
            mapdl.units("SI")
            mapdl.et(1, "PLANE42", kop3=3)
            mapdl.r(1, 0.001)  # thickness
            mapdl.mp("EX", 1, 210e9)
            mapdl.mp("DENS", 1, 7800)
            mapdl.mp("NUXY", 1, 0.3)

            # Create the main rectangle and circular notches
            rect_anum = mapdl.blc4(width=length, height=width)
            notch_anum = mapdl.cyl4(length / 2, 0, notch_radius)
            plate_with_notch_anum = mapdl.asba(rect_anum, notch_anum)
            second_notch_anum = mapdl.cyl4(length / 2, width, notch_radius)
            plate_with_notches_anum = mapdl.asba(plate_with_notch_anum, second_notch_anum)

            # Refine the mesh around the circular notch due to stress concentration
            notch_esize = np.pi * notch_radius / hole_ratio  # Element size around the notch
            plate_esize = width / plate_ratio  # Element size for the rest of the plate
            mapdl.lsel('S', 'LINE', '', 5)
            mapdl.lsel('A', 'LINE', '', 6)
            mapdl.lsel('A', 'LINE', '', 8)
            mapdl.lsel('A', 'LINE', '', 11)  # Update these values based on your notch configuration
            mapdl.lesize("ALL", notch_esize, kforc=1)
            mapdl.lsel("ALL")
            mapdl.esize(plate_esize)
            mapdl.amesh(plate_with_notches_anum)

            # Boundary Conditions
            mapdl.nsel("S", "LOC", "X", 0)
            left_nodes = np.array(mapdl.mesh.nodes)
            mapdl.d("ALL", "UX")
            mapdl.nsel("R", "LOC", "Y", width / 2)
            left_fixed_y_node = np.array(mapdl.mesh.nodes)
            assert mapdl.mesh.n_node == 1
            mapdl.d("ALL", "UY")
            mapdl.nsel("S", "LOC", "X", length)
            right_nodes = np.array(mapdl.mesh.nodes)
            assert np.allclose(mapdl.mesh.nodes[:, 0], length)
            mapdl.cp(5, "UX", "ALL")
            mapdl.nsel("R", "LOC", "Y", width / 2)
            right_force_y_node = np.array(mapdl.mesh.nodes)
            mapdl.f("ALL", "FX", force)
            mapdl.allsel(mute=True)

            # Solve the Static Problem
            start_time = time.time()
            mapdl.run("/SOLU")
            mapdl.antype("STATIC")
            output = mapdl.solve()
            end_time = time.time()
            run_time = end_time - start_time
            print(f"Run time for {refinement} sample {i + 1}: {run_time:.3f} seconds")

            # Post-Processing
            result = mapdl.result
            nnum, stress = result.principal_nodal_stress(0)
            von_mises = stress[:, -1]

            n_nodes = mapdl.mesh.n_node
            nodes = np.array(mapdl.mesh.nodes)
            elements = np.array(mapdl.mesh.elem)
            boundary_conditions = np.zeros((n_nodes, 7))

            # Apply boundary conditions
            distances = np.linalg.norm(nodes[:, None] - left_nodes, axis=2)
            indices_left_nodes = np.argmin(distances, axis=0)
            boundary_conditions[:, 0] = 1.0
            boundary_conditions[indices_left_nodes.tolist(), 0] = 0
            boundary_conditions[indices_left_nodes.tolist(), 1] = 1.0

            indices_left_fixed_y_node = np.where((nodes == left_fixed_y_node).all(axis=1))[0]
            boundary_conditions[:, 2] = 1.0
            boundary_conditions[indices_left_fixed_y_node, 2] = 0
            boundary_conditions[indices_left_fixed_y_node, 3] = 1.0

            distances = np.linalg.norm(nodes[:, None] - right_nodes, axis=2)
            indices_right_nodes = np.argmin(distances, axis=0)
            boundary_conditions[:, 4] = 1
            boundary_conditions[indices_right_nodes, 4] = 0
            boundary_conditions[indices_right_nodes, 5] = 1
            boundary_conditions[:, 6] = force

            # Prepare training data
            geometry = {
                'length': length,
                'width': width,
                'diameter': notch_diameter,
            }
            mesh = {
                'nodes': nodes[:, 0:-1],
                'element_nodes': elements[:, 10:14]
            }
            input_features = {
                'boundary_conditions': boundary_conditions,
            }
            target_labels = {
                'von_mises_stress': von_mises.tolist(),
            }

            training_data = [(geometry, mesh, input_features, target_labels)]

            # Load existing training data if available
            try:
                with open(file_name, 'rb') as f:
                    existing_data = pickle.load(f)
            except FileNotFoundError:
                existing_data = []

            existing_data += training_data

            # Save training data to a pickle file
            with open(file_name, 'wb') as f:
                pickle.dump(existing_data, f)

            print(f"Generated {refinement} training data for iteration {i + 1}/{num_samples}")

            mapdl.exit()

# Example usage
force_range = (500, 2000)  # Example force range (Newtons)
ratio_range = (0.2, 0.5)   # Example ratio range
num_samples = 10000        # Number of training samples to generate

generate_training_data(force_range, ratio_range, num_samples)
