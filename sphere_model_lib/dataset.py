import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import Chem
import numpy as np
import os

from_file_path = "../dataset/data.xlsx"
to_dir_path = "../arranged_dataset_1020"
for name in ["training", "test"]:
    df = pd.read_excel(from_file_path, sheet_name=name).dropna(subset=["smiles"])
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=["mol", "smiles"])
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    df["InchyKey"] = df["ROMol"].apply(Chem.MolToInchiKey)
    if name == "training":
        df = df[["entry", "InchyKey", "smiles", "ROMol", "dr.expt.BH3", "dr.expt.LiAlH4", "dr.expt.NaBH4",
                 "dr.expt.LiAl(OMe)3H", "dr.expt.MeLi", "dr.expt.MeMgI", "dr.expt.PhLi", "dr.expt.PhMgI"]]
    else:
        df = df[["entry", "InchyKey", "smiles", "ROMol"]]  # ,"dr.expt.NaBH4","dr.expt.MeLi"]]
        print(df)

    PandasTools.SaveXlsxFromFrame(df, "../dataset/data{}.xlsx".format(name), size=(100, 100))

os.makedirs(to_dir_path, exist_ok=True)


def make_dataset(sheet_name, column_name, out_file_name):  # in ["dr.expt.BH3"]:
    df = pd.read_excel(from_file_path, sheet_name=sheet_name) \
        .rename(columns={column_name: "dr.expt."}).dropna(subset=["smiles"])
    if True and sheet_name == "training":
        df = df[df["entry"] != 141]  # [df["entry"]!=86]
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = df[df["dr.expt."] != False].dropna(subset=['dr.expt.', "mol", "smiles"])  # .dropna(subset=['smiles'])#順番重要！

    df["RT"] = 1.99 * 10 ** -3 * df["temperature"].values
    df["ΔΔG.expt."] = df["RT"].values * np.log(100 / df["dr.expt."].values - 1)
    df = df[df["mol"].map(lambda mol:
                          not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][N,OH1,F,Cl,Br]"))
                          and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][#6]=O"))
                          and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*[N,OH1,F,Cl,Br]"))
                          and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*[#6]=O"))
                          and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*[#6]#N"))
                          and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]!-*"))
                          and not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6]*!-*"))
                          )]

    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    df = df[["entry", "smiles", "ROMol", "dr.expt.", "RT", "ΔΔG.expt."]]
    PandasTools.SaveXlsxFromFrame(df, to_dir_path + "/" + out_file_name, size=(100, 100))

    print(len(df))
    print("finish")


make_dataset("training", "dr.expt.BH3", "training_BH3.xls")
make_dataset("training", "dr.expt.NaBH4", "training_NaBH4.xls")
make_dataset("training", "dr.expt.LiAlH4", "training_LiAlH4.xls")
make_dataset("training", "dr.expt.LiAl(OMe)3H", "training_LiAl(OMe)3H.xls")
make_dataset("training", "dr.expt.MeLi", "training_MeLi.xls")
make_dataset("training", "dr.expt.MeMgI", "training_MeMgI.xls")
make_dataset("training", "dr.expt.PhLi", "training_PhLi.xls")
make_dataset("training", "dr.expt.PhMgI", "training_PhMgI.xls")
make_dataset("test", "dr.expt.LiAlH4", "test_LiAlH4.xls")
make_dataset("test", "dr.expt.NaBH4", "test_NaBH4.xls")
make_dataset("test", "dr.expt.MeLi", "test_MeLi.xls")
make_dataset("test", "dr.expt.PhLi", "test_PhLi.xls")
# make_dataset("test2", "dr.expt.EtMgI", "test_EtMgI.xls")
# make_dataset("test2", "dr.expt.iPrMgI", "test_iPrMgI.xls")
# make_dataset("test2", "dr.expt.tBuMgI", "test_tBuMgI.xls")
