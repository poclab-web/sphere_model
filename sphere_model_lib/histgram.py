import calculate_conformation
import os
import glob
import json
import pandas as pd
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
import matplotlib.pyplot as plt
from pylab import rcParams

# plt.hist(a-1, bins=100, alpha=0.3, histtype='stepfilled', color='r')
# plt.hist(a+1, bins=100, alpha=0.3, histtype='stepfilled', color='b')
# plt.show()
if __name__ == '__main__':
    if False:
        data_file_path = "../dataset/data.xlsx"
        sheet_name = "train"
        df1 = pd.read_excel(data_file_path, sheet_name=sheet_name).dropna(subset=['smiles'])
        sheet_name = "test"
        df2 = pd.read_excel(data_file_path, sheet_name=sheet_name).dropna(subset=['smiles'])
        df = pd.concat([df1, df2])
        df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
        df = df.dropna(subset=['mol'])
        df["molwt"] = df["smiles"].apply(lambda smiles: ExactMolWt(Chem.MolFromSmiles(smiles)))
        df = df.sort_values("molwt")  # [:2]
    else:
        datafilename = "../arranged_dataset_1020/*.xls"
        l = glob.glob(datafilename)
        dfs = []
        for name in l:
            df = pd.read_excel(name).dropna(subset=['smiles'])
            print(name, len(df))
            #df["mol"] = df["smiles"].apply(get_mol)
            dfs.append(df)
        df = pd.concat(dfs).drop_duplicates(subset=["smiles"])#.dropna(subset=['mol'])
    print(len(df))

    for param_file_name in glob.glob("../parameter/single_point_calculation_parameter/*.txt"):
        with open(param_file_name, "r") as f:
            param = json.loads(f.read())

        for name in ["LUMO","Dt"]:
            ab = []
            bc = []
            others = []
            rcParams['figure.figsize'] = 5, 5
            plt.figure()
            for smiles in df["smiles"]:  # [:100]:
                mol = calculate_conformation.get_mol(smiles)
                input_dirs_name = param["psi4_aligned_save_file"] + "/" + mol.GetProp("InchyKey")
                output_dirs_name = param["one_point_out_file"] + "/" + mol.GetProp("InchyKey")
                print(output_dirs_name)
                i=0
                while os.path.isfile("{}/data{}.pkl".format(output_dirs_name, i)):

                    df_ = pd.read_pickle("{}/data{}.pkl".format(output_dirs_name, i))
                    if name=="Dt":
                        plt.hist(df_[name],range=[0,100], bins=100, alpha=0.02, histtype='stepfilled', log=True, color='r')
                    else:
                        plt.hist(df_[name], bins=100, alpha=0.02, histtype='stepfilled', log=True, color='r')
                    i+=1
                # df[["LUMO","Dt"]].hist(bins=100, histtype='stepfilled',log=True)
            if name=="Dt":
                plt.xlabel('electron density [e/$\mathrm{Bohr^3}$]')
            else:
                plt.xlabel('LUMO [e/$\mathrm{Bohr^{3/2}}$]')

            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig("../figs/all/hist_{}_{}.png".format(name,param["one_point_level"].replace("/","_")), dpi=300)
