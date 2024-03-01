import psi4
import os
import glob
import numpy as np

psi4.set_options({'geom_maxiter': 1000})
opt_level = "b3lyp/6-31g(d)"
level = "b3lyp/6-311g(d,p)"
input_dir_name = "../reactant_xyz"
output_dir_name = "../reactant_xyz_out/" + opt_level.replace("/", "_")
single_dir_name = "../reactant_xyz_out/" + opt_level.replace("/", "_") + level.replace("/", "_")
for input_file_name in os.listdir(input_dir_name):  # 3,5,9["CH3-.xyz",'OH-.xyz','BH4-.xyz']:#
    name = os.path.splitext(input_file_name)[0]
    print(name)
    output_dir_name_local = output_dir_name + "/" + name
    os.makedirs(output_dir_name_local, exist_ok=True)
    single_dir_name_local = single_dir_name + "/" + name
    os.makedirs(single_dir_name_local, exist_ok=True)
    psi4.set_options({'cubeprop_filepath': output_dir_name_local})
    try:
        file = "{}/optimized.xyz".format(output_dir_name_local)
        if not os.path.isfile(file):
            with open(input_dir_name + "/" + input_file_name, "r") as f:
                rl = f.read().split("\n")
                mol_input = "0 1\n nocom\n noreorient\n " + "\n".join(rl[2:])
                molecule = psi4.geometry(mol_input)
            energy = psi4.optimize(opt_level, molecule=molecule)
            print(file)
            open(file, 'w')
            with open(file, 'w') as f:
                print(molecule.natom(), file=f)
                print(energy * psi4.constants.hartree2kcalmol, file=f)
                print("\n".join(molecule.save_string_xyz().split('\n')[1:]), file=f)
        # mol = Chem.MolFromXYZFile(file)
        # mol = AllChem.DeleteSubstructs(mol, Chem.MolFromSmiles("[MgH2]"))
        # mol = AllChem.DeleteSubstructs(mol, Chem.MolFromSmiles("Br"))
        # mol = AllChem.DeleteSubstructs(mol, Chem.MolFromSmiles("[NaH]"))
        # mol = AllChem.DeleteSubstructs(mol, Chem.MolFromSmiles("[LiH]"))
        # print(Chem.MolToSmiles(mol))
        # # print(Chem.MolToMolBlock(mol))
        # volume = AllChem.ComputeMolVolume(mol)
        # r = (volume / np.pi * 3 / 4) ** (1 / 3)
        # print(r)

        if not os.path.isfile(output_dir_name_local + "/Dt.cube"):
            with open(file, "r") as f:
                rl = f.read().split("\n")
                mol_input = "0 1\n nocom\n noreorient\n " + "\n".join(rl[2:])
                molecule = psi4.geometry(mol_input)

            energy, wfn = psi4.energy(level, molecule=molecule, return_wfn=True)
            psi4.set_options({'cubeprop_tasks': ['frontier_orbitals', "esp"],
                              "cubic_grid_spacing": [0.2, 0.2, 0.2],
                              "cubeprop_isocontour_threshold": 0.99,
                              "cubic_grid_overage": [10, 10, 10]})
            psi4.cubeprop(wfn)
            HOMO_file_name = glob.glob(output_dir_name_local + "/Psi_a_*-A_HOMO.cube")[0]
            with open(HOMO_file_name, "r") as f:
                rl = f.read().split("\n")
            with open(HOMO_file_name, "w") as f:
                print(rl[0], file=f)
                print(rl[1] + " {}".format(wfn.epsilon_a_subset("AO", "ALL").np[wfn.nalpha() - 1]), file=f)
                print("\n".join(rl[2:]), file=f)
            LUMO_file_name = glob.glob(output_dir_name_local + "/Psi_a_*-A_LUMO.cube")[0]
            with open(LUMO_file_name, "r") as f:
                rl = f.read().split("\n")
            with open(LUMO_file_name, "w") as f:
                print(rl[0], file=f)
                print(rl[1] + " {}".format(wfn.epsilon_a_subset("AO", "ALL").np[wfn.nalpha()]), file=f)
                print("\n".join(rl[2:]), file=f)
        file = "{}/frequencies.txt".format(output_dir_name_local)
        if not os.path.isfile(file):
            energy, wfn = psi4.frequency(level,
                                         molecule=molecule,
                                         return_wfn=True)
            with open(file, 'w') as f:
                print(np.array(wfn.frequencies()), file=f)
    except Exception as e:
        print(e)
        continue
