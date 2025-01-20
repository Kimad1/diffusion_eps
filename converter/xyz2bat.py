# converters/xyz2bat.py

import os
import logging
import numpy as np
import networkx as nx
import mdtraj as md
from pathlib import Path
from typing import List, Tuple

class XYZ2BATconverter:
    """
    Converter class to transform XYZ files into BAT format based on bond information.                                    #converts XYZ files into Bond, Angle, Torsion format

    Args:
        ts_file (str): Path to the transition state PDB file.
        nb1 (int): First atom number in bond 1.
        nb2 (int): Second atom number in bond 1.
    """
    def __init__(self, ts_file: str, nb1: int, nb2: int):
        self.reference = md.load(ts_file)                                                                                 #Path to transition state PDB file is loaded as a reference structure using MDTraj(read, write, analyze molecular dynammic trajectories)
        self.nb1 = nb1
        self.nb2 = nb2
        self.graph = self._generate_connectivity()                                                                        #Builds a connectivity graph based on the bonds (helps determine topology)
        self.atom_count, self.topology = self._generate_topology()                                                        #Determines the number of atoms and list of paths.
        logging.info("XYZ2BATconverter initialized.")                                                                     #Log initialization of the converter (to help debug or track the execution)

    def _generate_connectivity(self) -> dict:                                                                             #Generates a dictionary (dict)
        """
        Generate connectivity graph from the transition state structure.

        Returns:
            dict: Connectivity graph with atom indices as keys and connected atom indices as values.
        """
        graph = {}                                                                                                         #Empties the dictionary
        for bond in self.reference.top.bonds:                                                                              #Loops through all the bonds in the structure...
            index1, index2 = bond.atom1.index, bond.atom2.index                                                            #Gets the index (positions) of both atom 1 and atom 2 that forms a bond.
            graph.setdefault(index1, []).append(index2)                                                                    #Adds index 2 to the list of neighbors for index 1. If index 1 doesn't exist as a key in graph, the set default puts an empty list, and adds index 2 to it.
            graph.setdefault(index2, []).append(index1)                                                                    #Does the same thing but for index 1. This makes sure the bond is bidirectional.

        # Ensure bond between nb1 and nb2 exists
        graph.setdefault(self.nb1 - 1, []).append(self.nb2 - 1)                                                            #Adds a connection between nb1 and nb2. (-1 for zerobased adjusting) This ensures a bond exists between the two atoms.
        graph.setdefault(self.nb2 - 1, []).append(self.nb1 - 1)                                                            #Same for the other way around. Ensures the symmetry.

        logging.debug("Connectivity graph generated.")                                                                     #Log initialization (to help debug or track execution)
        return graph                                                                                                       #You get the graph containing the atom 1-atom 2 connectivity information.

    def _generate_topology(self) -> Tuple[int, List[Tuple[int, ...]]]:                                                     #Function returns the number of atoms (int) and a list of the unique paths of atoms based on the connectivity
        """
        Generate topology paths based on connectivity.

        Returns:
            Tuple[int, List[Tuple[int, ...]]]: Number of atoms and list of DOF paths.
        """
        G = nx.Graph(self.graph)                                                                                           #Creates a graph (network) using the connections from self.graph. (dots: atoms, lines: connections between atoms)
        all_paths = [                                                                                                      #Finds all possible paths
            path for i in range(len(self.graph)) for j in range(len(self.graph))                                           #Starts with the first atom (i) and goes through every atom in the graph. | j (other atom) 
            for path in nx.all_simple_paths(G, source=i, target=j) if len(path) in [2, 3, 4]                               #Finds all possible paths between i and j but only the paths that has a length of 2,3,4 atoms.
        ]

        unique_paths = []                                                                                                  #Creates an empty list
        for items in all_paths:                                                                                            #Loops through each item in all_paths
            path_tuple = tuple(items)                                                                                      #Makes the list into a unchangeable list. [] --> ()
            if path_tuple not in unique_paths and tuple(reversed(path_tuple)) not in unique_paths:                         #Checks if path_tuple list and its reversed version is not already in unique_paths
                unique_paths.append(path_tuple)                                                                            #If it isn't already in it, it adds it to the unique_paths list.
        unique_paths.sort()                                                                                                #Sorts the paths into ascending order. (3,2,1) --> (1,2,3) 
        unique_paths.sort(key=len)                                                                                         #Sorts the paths by length (shorter list comes first.) 

        logging.info(f"Generated topology with {len(unique_paths)} paths.")                                                #Generated topology with # of paths paths.
        return len(self.graph), unique_paths                                                                               #Returns the total number of atoms in the graph (len(self.graph) and the list (unique_paths)

    def save_topology(self, output_dir: Path) -> None:                                                                     #Defines save_topology (allows access to self,and where it should be saved) no output More of an action
        """
        Save the generated topology to a file.

        Args:
            output_dir (Path): Directory to save the topology file.
        """
        output_path = output_dir / "topology.txt"                                                                          #Places the outpath path in the topology.txt file.
        try:                                                                                                               #try-except block (handles potential errors)
            with open(output_path, "w") as file:                                                                           #opens the output_path in write mode.
                file.write(f"{self.atom_count}\n")                                                                         #writes the total number of atoms in the first line of the file. \n stars a new line
                for path in self.topology:                                                                                 #Loops through path item in self.topology (list of atom paths)
                    file.write(" ".join(str(atom + 1) for atom in path) + "\n")                                            #Writes each path to the file: Joins elements of a list/ converts each atom index to a string with +1 added so it's easier to read (starts at 1 instead of 0) This repeats for every atom and starts a new line every time.
            logging.info(f"Topology saved to {output_path}")                                                               #Saying if it's successful, "Topology saved to _specified file_"
        except Exception as e:                                                                                             #If there is an error (e) within the file...
            logging.error(f"Failed to save topology: {e}")                                                                 #Says if file-saving operation failed, "Failed to save topology: details of the error"

    def process_xyz_files(self, output_dir: Path, ts: str, atom1: int, atom2: int) -> None:                                #defines the function (performs an action/no output)
        """
        Process XYZ files to compute internal coordinates and save DOFs.

        Args:
            output_dir (Path): Directory containing generated XYZ files.
            ts (str): Transition state PDB file name.
            atom1 (int): First atom number in reacting bond.
            atom2 (int): Second atom number in reacting bond.
        """
        trajectory = md.load_xyz(output_dir / "snapshot.xyz", top=ts)                                                      #Loads the snapshot.xyz file in output_dir. top specifies topology information from ts file. (transition state)
        self.topology.insert(0, (atom1 - 1, atom2 - 1))  # Add reacting bond to topology                                   #Inserts the reacting bond list (atom1 - 1, atom2 - 1) (zero indexed (atom numbers 1 based in molecular data) to the beginning of topology list.

        # Separate DOFs based on their lengths
        bond_list = [path for path in self.topology if len(path) == 2]                                                     #
        angle_list = [path for path in self.topology if len(path) == 3]                                                    #
        torsion_list = [path for path in self.topology if len(path) == 4]                                                  #
                                            
        # Compute internal coordinates
        b = md.compute_distances(trajectory, bond_list) * 10.0  # nm to angstrom                                           #
        a = md.compute_angles(trajectory, angle_list, periodic=True)  # radians                                            #
        t = md.compute_dihedrals(trajectory, torsion_list, periodic=True)  # radians                                       #

        # Concatenate all DOFs
        dofs = np.hstack((b, a, t))                                                                                        #
        np.save(output_dir / "dof.npy", dofs)                                                                              #
        logging.info(f"DOFs saved to {output_dir / 'dof.npy'}")                                                            #

