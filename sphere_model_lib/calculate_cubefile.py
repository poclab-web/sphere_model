import json
import pandas as pd
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
import calculate_conformation
import psi4
import os
import glob
import numpy as np
from itertools import product
import time


def psi4calculation(input_dir_name, output_dir_name, level="hf/sto-3g"):
    psi4.set_num_threads(nthread=6)
    psi4.set_memory("10GB")
    # psi4.set_options({'geom_maxiter': 1000})

    psi4.set_options({'cubeprop_filepath': output_dir_name})
    print(input_dir_name)
    # i = 0
    # while os.path.isfile("{}/optimized{}.xyz".format(input_dir_name, i)):
    for filepath in sorted(glob.glob("{}/optimized?.xyz".format(input_dir_name))):
        i = filepath[-5]
        os.makedirs(output_dir_name, exist_ok=True)
        with open(filepath, "r") as f:
            rl = f.read().split("\n")
            mol_input = "0 1\n nocom\n noreorient\n " + "\n".join(rl[2:])
            molecule = psi4.geometry(mol_input)
            input_energy = rl[1]
        energy, wfn = psi4.energy(level, molecule=molecule, return_wfn=True)
        psi4.set_options({'cubeprop_tasks': ['frontier_orbitals', 'esp', 'dual_descriptor', 'orbitals'],
                          'cubeprop_orbitals': [wfn.nalpha() + 2],
                          "cubic_grid_spacing": [0.2, 0.2, 0.2],
                          "cubeprop_isocontour_threshold": 0.99,
                          "cubic_grid_overage": [5, 5, 10]})
        psi4.cubeprop(wfn)
        os.rename(glob.glob(output_dir_name + "/Psi_a_*_LUMO.cube")[0], output_dir_name + "/LUMO{}.cube".format(i))
        os.rename(glob.glob(output_dir_name + "/Psi_a_*_HOMO.cube")[0], output_dir_name + "/HOMO{}.cube".format(i))

        with open(output_dir_name + "/HOMO{}.cube".format(i), "r") as f:
            rl = f.read().split("\n")
        with open(output_dir_name + "/HOMO{}.cube".format(i), "w") as f:
            print(rl[0], file=f)
            print(rl[1] + " {}".format(wfn.epsilon_a_subset("AO", "ALL").np[wfn.nalpha() - 1]), file=f)
            print("\n".join(rl[2:]), file=f)
        with open(output_dir_name + "/LUMO{}.cube".format(i), "r") as f:
            rl = f.read().split("\n")
        with open(output_dir_name + "/LUMO{}.cube".format(i), "w") as f:
            print(rl[0], file=f)
            print(rl[1] + " {}".format(wfn.epsilon_a_subset("AO", "ALL").np[wfn.nalpha()]), file=f)
            print("\n".join(rl[2:]), file=f)

        os.rename(glob.glob(output_dir_name + "/Psi_a_*.cube")[0], output_dir_name + "/LUMO+1{}.cube".format(i))
        with open(output_dir_name + "/LUMO+1{}.cube".format(i), "r") as f:
            rl = f.read().split("\n")
        with open(output_dir_name + "/LUMO+1{}.cube".format(i), "w") as f:
            print(rl[0], file=f)
            print(rl[1] + " {}".format(wfn.epsilon_a_subset("AO", "ALL").np[wfn.nalpha() + 1]), file=f)
            print("\n".join(rl[2:]), file=f)

        os.rename(output_dir_name + "/Dt.cube", output_dir_name + "/Dt{}.cube".format(i))
        os.rename(output_dir_name + "/ESP.cube", output_dir_name + "/ESP{}.cube".format(i))

        os.rename(glob.glob(output_dir_name + "/DUAL_*.cube")[0], output_dir_name + "/DUAL{}.cube".format(i))
        with open(output_dir_name + "/geom.xyz", "r") as f:
            rl = f.read().split("\n")
            mol_output = rl[0] + "\n" + input_energy + "\n" + "\n".join(rl[2:])
        os.remove(output_dir_name + "/geom.xyz")
        with open(output_dir_name + "/optimized{}.xyz".format(i), "w") as f:
            print(mol_output, file=f)
        # i += 1


def cube_to_pkl(dirs_name):
    # i = 0
    # while os.path.isfile("{}/optimized{}.xyz".format(dirs_name + "calculating", i)):
    for filepath in sorted(glob.glob("{}/optimized?.xyz".format(dirs_name + "calculating"))):
        i = filepath[-5]
        with open("{}/Dt{}.cube".format(dirs_name + "calculating", i), 'r', encoding='UTF-8') as f:
            Dt = f.read().splitlines()
        with open("{}/ESP{}.cube".format(dirs_name + "calculating", i), 'r', encoding='UTF-8') as f:
            ESP = f.read().splitlines()
        with open("{}/LUMO{}.cube".format(dirs_name + "calculating", i), 'r', encoding='UTF-8') as f:
            LUMO = f.read().splitlines()
        with open("{}/LUMO+1{}.cube".format(dirs_name + "calculating", i), 'r', encoding='UTF-8') as f:
            LUMO1 = f.read().splitlines()
        with open("{}/HOMO{}.cube".format(dirs_name + "calculating", i), 'r', encoding='UTF-8') as f:
            HOMO = f.read().splitlines()
        with open("{}/DUAL{}.cube".format(dirs_name + "calculating", i), 'r', encoding='UTF-8') as f:
            DUAL = f.read().splitlines()

        l = np.array([_.split() for _ in Dt[2:6]])
        n_atom = int(l[0, 0])
        x0 = l[0, 1:].astype(float)
        size = l[1:, 0].astype(int)
        axis = l[1:, 1:].astype(float)
        Dt = np.concatenate([_.split() for _ in Dt[3 + 3 + n_atom:]]).astype(float).reshape(-1, 1)
        ESP = np.concatenate([_.split() for _ in ESP[3 + 3 + n_atom:]]).astype(float).reshape(-1, 1)
        LUMO = np.concatenate([_.split() for _ in LUMO[3 + 3 + n_atom:]]).astype(float).reshape(-1, 1)
        LUMO1 = np.concatenate([_.split() for _ in LUMO1[3 + 3 + n_atom:]]).astype(float).reshape(-1, 1)
        HOMO = np.concatenate([_.split() for _ in HOMO[3 + 3 + n_atom:]]).astype(float).reshape(-1, 1)
        DUAL = np.concatenate([_.split() for _ in DUAL[3 + 3 + n_atom:]]).astype(float).reshape(-1, 1)
        l = np.array(list(product(range(size[0]), range(size[1]), range(size[2])))) @ axis + x0
        l = l * psi4.constants.bohr2angstroms
        arr = np.concatenate([l, Dt, ESP, LUMO, LUMO1, HOMO, DUAL], axis=1)
        df = pd.DataFrame(arr, columns=["x", "y", "z", "Dt", "ESP", "LUMO", "LUMO+1", "HOMO", "DUAL"])
        df.to_pickle(dirs_name + "calculating" + "/data{}.pkl".format(i))
    try:
        print(i)
        os.rename(dirs_name + "calculating", dirs_name)
    except:
        None


if __name__ == '__main__':
    datafilename = "../arranged_dataset/*.xls"
    l = glob.glob(datafilename)
    print(l)
    dfs = []
    for name in l:
        df = pd.read_excel(name).dropna(subset=['smiles'])
        print(name, len(df))
        df["mol"] = df["smiles"].apply(calculate_conformation.get_mol)
        dfs.append(df)
    df = pd.concat(dfs).drop_duplicates(subset=["smiles"]).dropna(subset=['mol'])
    df["molwt"] = df["smiles"].apply(lambda smiles: ExactMolWt(Chem.MolFromSmiles(smiles)))
    df = df.sort_values("molwt")  # [:2]
    df["InchyKey"] = [mol.GetProp("InchyKey") for mol in df["mol"]]
    print(len(df))
    print(len(df.duplicated(subset='InchyKey')))
    while True:
        # time.sleep(3600 * 48)
        for param_file_name in glob.glob("../parameter/single_point_calculation_parameter/*.txt"):
            with open(param_file_name, "r") as f:
                param = json.loads(f.read())
            ab = []
            bc = []
            others = []
            for i, smiles in enumerate(df["smiles"]):
                mol = calculate_conformation.get_mol(smiles)
                if mol.GetProp("InchyKey") in ["MILHJIWCSVKZDK-NAKRPEOUSA-N",
                                               "ASNHUYVMPRNXNB-NAKRPEOUSA-N",
                                               "ZALGHXJCZDONDI-XNRSKRNUSA-N"]:

                    input_dirs_name = param["psi4_aligned_save_file"] + "/" + mol.GetProp("InchyKey") + "UFF"
                    output_dirs_name = param["one_point_out_file"] + "/" + mol.GetProp("InchyKey") + "UFF"
                    if not os.path.isdir(output_dirs_name):
                        psi4calculation(input_dirs_name, output_dirs_name + "calculating", param["one_point_level"])
                        cube_to_pkl(output_dirs_name)
                input_dirs_name = param["psi4_aligned_save_file"] + "/" + mol.GetProp("InchyKey")
                output_dirs_name = param["one_point_out_file"] + "/" + mol.GetProp("InchyKey")
                if not os.path.isdir(output_dirs_name):
                    psi4calculation(input_dirs_name, output_dirs_name + "calculating", param["one_point_level"])
                    cube_to_pkl(output_dirs_name)
            print(i, output_dirs_name, smiles)

        time.sleep(10)
