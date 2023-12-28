import pandas as pd
import calculate_conformation
import numpy as np
import os
import time
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut
from rdkit.Chem import PandasTools
import json
import glob


def pkl_to_featurevalue(dirs_name, dfp, mol, param):
    dfp = dfp.sort_values(by="r", ascending=False)
    i = 0
    while os.path.isfile("{}/data{}.pkl".format(dirs_name, i)):
        data = pd.read_pickle("{}/data{}.pkl".format(dirs_name, i))
        data = data[(data["x"] > dfp["x"].min() - dfp["r"].max())]
        data = data[(data["y"] > dfp["y"].min() - dfp["r"].max())]
        data = data[(data["y"] < dfp["y"].max() + dfp["r"].max())]
        data = [data[data["z"] > 0], data[data["z"] < 0]]
        coordinate = [data[0][["x", "y", "z"]].to_numpy(), data[1][["x", "y", "z"]].to_numpy()]
        for cent in dfp[["x", "y", "z"]].drop_duplicates().values:
            dfp_ = dfp[(dfp["x"] == cent[0]) & (dfp["y"] == cent[1]) & (dfp["z"] == cent[2])]
            d = [np.linalg.norm(coordinate[0] - np.array(cent), axis=1),
                 np.linalg.norm(coordinate[1] - np.array(cent) * np.array([1, -1, -1]), axis=1)]
            cond = list(map(lambda _: _ < dfp_["r"].max(), d))
            d_ = [d[0][cond[0]], d[1][cond[1]]]
            data_ = [data[0][cond[0]], data[1][cond[1]]]
            for p, r in zip(dfp_[["r", "d", "t"]].values, dfp_["r"]):
                data__ = [data_[0][d_[0] < r], data_[1][d_[1] < r]]
                features = {}
                dr = (0.52917720859 * 0.2) ** 3
                if "Dt" in param:
                    isoval = 10
                    Dts = [(np.sum(
                        np.where(data__[0]["Dt"].to_numpy() < isoval, data__[0]["Dt"].to_numpy(), isoval))) * dr,
                           (np.sum(
                               np.where(data__[1]["Dt"].to_numpy() < isoval, data__[1]["Dt"].to_numpy(), isoval))) * dr]
                    features["Dt"] = Dts
                if "LUMO" in param:
                    LUMO_up = data__[0]["LUMO"].to_numpy()
                    LUMO_down = data__[1]["LUMO"].to_numpy()
                    LUMOs = [(np.sum(LUMO_up) * dr) ** 2, (np.sum(LUMO_down) * dr) ** 2]
                    LUMO1_up = data__[0]["LUMO+1"].to_numpy()
                    LUMO1_down = data__[1]["LUMO+1"].to_numpy()
                    LUMO1s = [(np.sum(LUMO1_up) * dr) ** 2, (np.sum(LUMO1_down) * dr) ** 2]
                    features["LUMO (calculated value)"] = LUMOs
                    features["LUMO+1"] = LUMO1s
                    if np.sum(LUMOs) > np.sum(LUMO1s) * 0.002:
                        features["LUMO"] = LUMOs
                    else:
                        features["LUMO"] = LUMO1s

                if "DUAL" in param:
                    DUAL_up = data__[0]["DUAL"].to_numpy()
                    DUAL_down = data__[1]["DUAL"].to_numpy()
                    DUALs = [np.sum(DUAL_up) * dr, np.sum(DUAL_down) * dr]
                    features["DUAL"] = DUALs
                if "ESP" in param:
                    ESP_up = data__[0]["ESP"].to_numpy()
                    ESP_down = data__[1]["ESP"].to_numpy()
                    isoval = 0.01
                    ESPs = [np.sum(np.where(np.abs(ESP_up) < isoval, ESP_up, isoval * np.sign(ESP_up))) * dr,
                            np.sum(np.where(np.abs(ESP_down) < isoval, ESP_down, isoval * np.sign(ESP_down))) * dr]
                    features["ESP"] = ESPs
                mol.GetConformer(i).SetProp("features_list," + ",".join(map(str, p)), json.dumps(features))
        i += 1


def energy_to_Boltzmann_distribution(mol, RT=1.99e-3 * 273):
    energies = np.array([float(conf.GetProp("energy")) for conf in mol.GetConformers()])
    energies = energies - np.min(energies)
    rates = np.exp(-energies / RT)
    rates = rates / np.sum(rates)
    for conf, rate in zip(mol.GetConformers(), rates):
        conf.SetProp("Boltzmann_distribution", str(rate))


def feature_value(mol, dfp, param):
    for p in dfp[["r", "d", "t"]].values:
        weights = [float(conf.GetProp("Boltzmann_distribution")) for conf in mol.GetConformers()]
        lis = mol.GetConformers()

        for name in param:
            l=[]
            for conf in lis:
                line = json.loads(conf.GetProp("features_list," + ",".join(map(str, p))))
                l.append(line[name])
            l = np.array(l).astype(float)

            if True and name == "Dt":
                w=np.exp(-l/np.sqrt(np.average(l**2))
                         )*np.array(weights).reshape(-1,1)
                ans=np.average(l,weights=w,axis=0)
                ans=ans[0]-ans[1]
            else:
                l=l[:,0]-l[:,1]
                ans = np.average(l, weights=weights)
            mol.SetProp(name + "," + ",".join(map(str, p)), str(ans))

def grid_search(df, dfp, param, output_dir_name="../results/test", output_file_name="/grid_search.csv"):
    RMSEs = []
    coefs = []
    stds = []
    r2s = []
    for p in dfp[["r", "d", "t"]].values:
        l_normalized = []
        l = []
        for name in param:
            feature = np.array([float(mol.GetProp(name + "," + ",".join(map(str, p)))) for mol in df["mol"]])
            feature_normalized = feature / np.sqrt(np.average(feature ** 2))
            if name in ["Dt"]:
                l_normalized.append(feature_normalized)
                l.append(feature)
            if name in ["LUMO", "DUAL", "ESP"]:
                l_normalized.append(-feature_normalized)
                l.append(feature)
        value_normalized = np.stack(l_normalized).T
        value = np.stack(l).T
        model = linear_model.LinearRegression(fit_intercept=False, positive=True).fit(value_normalized, df["ΔΔG.expt."])
        predict = model.predict(value_normalized).T
        predict = np.where(np.abs(predict) < 2.5, predict, 2.5 * np.sign(predict))
        r2 = r2_score(df["ΔΔG.expt."], predict)
        r2s.append(r2)
        RMSE = np.sqrt(mean_squared_error(df["ΔΔG.expt."], predict))
        RMSEs.append(RMSE)
        coef = model.coef_.tolist()
        coefs.append(coef)
        stds.append(np.sqrt(np.average(value ** 2, axis=0)))
    dfp["RMSE"] = RMSEs
    dfp["r2"] = r2s

    for i, name in enumerate(param):

        if name in ["LUMO", "DUAL", "ESP"]:
            dfp["b_"+name] = -np.array(coefs)[:, i]
            dfp[name + "_std"] = np.array(stds)[:, i]
        if name in ["Dt"]:
            dfp["b_electron"] = np.array(coefs)[:, i]
            dfp["electron_std"] = np.array(stds)[:, i]
    os.makedirs(output_dir_name, exist_ok=True)
    dfp.to_csv(output_dir_name + output_file_name)
    return dfp


def leave_one_out(df, p, param, output_dir_name="../results/test", output_file_name="/leave_one_out.xls"):
    df["features"] = [[conf.GetProp("features_list," + ",".join(map(str, p))) for conf in mol.GetConformers()] for mol
                      in df["mol"]]
    l = []
    for name in param:
        feature = np.array(
            [float(mol.GetProp(name + "," + ",".join(map(str, p)))) for mol in df["mol"]])
        feature_normalized = feature / np.sqrt(np.average(feature ** 2))

        if name in ["Dt"]:
            df["normalized electron"] = feature_normalized
            df["electron"] = feature
            l.append(feature)
        if name in ["LUMO", "DUAL", "ESP"]:
            df["normalized "+name] = feature_normalized
            df[name] = feature
            l.append(-feature)
    value = np.stack(l).T
    model = linear_model.LinearRegression(fit_intercept=False, positive=True).fit(value, df["ΔΔG.expt."])
    predict = model.predict(value).T
    df["ΔΔG.regression"] = np.where(np.abs(predict) < 2.5, predict, 2.5 * np.sign(predict))
    df["dr.regression"] = 100 / (1 / np.exp(-df["ΔΔG.regression"] / df["RT"]) + 1)

    for i, name in enumerate(param):
        if name in ["Dt"]:
            df["electron contribution"] = df["electron"] * model.coef_[i]
        if name in ["LUMO", "DUAL", "ESP"]:
            df[name + " contribution"] = -df[name] * model.coef_[i]

    predicts = []
    for (train_index, test_index) in LeaveOneOut().split(df):
        l = []
        for name in param:
            feature = np.array(
                [float(mol.GetProp(name + "," + ",".join(map(str, p)))) for mol in df.iloc[train_index]["mol"]])
            if name in ["Dt"]:
                l.append(feature)
            if name in ["LUMO", "DUAL", "ESP"]:
                l.append(-feature)
        value = np.stack(l).T
        model_ = linear_model.LinearRegression(fit_intercept=False, positive=True).fit(value,
                                                                                      df.iloc[train_index]["ΔΔG.expt."])
        l = []
        for name in param:
            feature = np.array(
                [float(mol.GetProp(name + "," + ",".join(map(str, p)))) for mol in df.iloc[test_index]["mol"]])
            if name in ["Dt"]:
                l.append(feature)
            if name in ["LUMO", "DUAL", "ESP"]:
                l.append(-feature)
        value = np.stack(l).T
        predict = model_.predict(value).T
        predict = np.where(np.abs(predict) < 2.5, predict, 2.5 * np.sign(predict))
        predicts.extend(predict)
    df["ΔΔG.loo"] = predicts
    df["InchyKey"] = df["mol"].apply(lambda mol: mol.GetProp("InchyKey"))
    df["error"] = df["ΔΔG.loo"] - df["ΔΔG.expt."]
    try:
        df=df.drop(["mol","train"], axis='columns')

    except:
        None

    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    PandasTools.SaveXlsxFromFrame(df, output_dir_name + output_file_name)  # , size=(100, 100))
    return model


def prediction(model, df_test, p, param, output_dir_name="../results/test", output_file_name="/train_test.xls"):
    df_test["features"] = [[conf.GetProp("features_list," + ",".join(map(str, p))) for conf in mol.GetConformers()] for mol
                      in df_test["mol"]]
    l = []
    for name in param:
        feature = np.array([float(mol.GetProp(name + "," + ",".join(map(str, p)))) for mol in df_test["mol"]])
        df_test[name] = feature
        if name in ["Dt"]:
            l.append(feature)
        if name in ["LUMO", "DUAL", "ESP"]:
            l.append(-feature)
    value = np.stack(l).T
    for i, name in enumerate(param):
        if name in ["Dt"]:
            df_test["electron contribution"] = df_test[name] * model.coef_[i]
        if name in ["LUMO", "DUAL", "ESP"]:
            df_test[name + " contribution"] = -df_test[name] * model.coef_[i]
    predict = model.predict(value).T
    df_test["ΔΔG.predict"] = np.where(np.abs(predict) < 2.5, predict, 2.5 * np.sign(predict))
    df_test["error"] = df_test["ΔΔG.predict"] - df_test["ΔΔG.expt."]
    df_test["InchyKey"] = df_test["mol"].apply(lambda mol: mol.GetProp("InchyKey"))
    df_test = df_test.drop(["mol", "train"], axis='columns')
    PandasTools.AddMoleculeColumnToFrame(df_test, "smiles")
    os.makedirs(output_dir_name, exist_ok=True)
    PandasTools.SaveXlsxFromFrame(df_test, output_dir_name + output_file_name)  # , size=(100, 100))


if __name__ == '__main__':
    for param_file_name in glob.glob("../parameter/run_sphere_model_parameter_1120/*.txt"):
        with open(param_file_name, "r") as f:
            param = json.loads(f.read())
        print(param)
        datafilename="../arranged_dataset_1020/*.xls"
        l=glob.glob(datafilename)
        print(l)
        all_names=[]
        dfs=[]
        for name in l:
            df=pd.read_excel(name).dropna(subset=['smiles'])#[:3]
            print(name,len(df))
            df["mol"] = df["smiles"].apply(calculate_conformation.get_mol)
            df["train"]="train" in os.path.basename(name)[0]
            all_names.append(os.path.splitext(os.path.basename(name))[0])
            df = df[
                [os.path.isdir(param["one_point_out_file"] + "/" + mol.GetProp("InchyKey")) for mol in df["mol"]]]
            dfs.append(df)
        df = pd.concat(dfs).drop_duplicates(subset=["smiles"])
        print(all_names)
        dfs = dict(zip(all_names, dfs))
        print(len(df))
        df["mol"].apply(
            lambda mol: calculate_conformation.read_xyz(mol,
                                                        param["one_point_out_file"] + "/" + mol.GetProp("InchyKey")))

        dfp = pd.read_csv(param["sphere_parameter_path"])
        print(len(dfp))

        start = time.time()
        for mol in df["mol"]:
            dirs_name = param["one_point_out_file"] + "/" + mol.GetProp("InchyKey")
            pkl_to_featurevalue(dirs_name, dfp, mol, param["feature"])

        for name in ["BH3", "NaBH4", "LiAlH4", "LiAl(OMe)3H", "MeLi", "MeMgI", "PhLi", "PhMgI"]:
            df_ = dfs["train_" + name]
            df_["mol"] = df_["smiles"].apply(lambda smiles: df[df["smiles"] == smiles]["mol"].iloc[0])
            for mol, RT in zip(df_["mol"], df_["RT"]):
                energy_to_Boltzmann_distribution(mol, RT)
                feature_value(mol, dfp, param["feature"])

            dfp_ = grid_search(df_, dfp, param["feature"], "{}/{}".format(param["save_dir"], name), "/grid_search.csv")
            model=leave_one_out(df_, dfp_[["r", "d", "t"]].values[dfp_["RMSE"].idxmin()], param["feature"],
                          "{}/{}".format(param["save_dir"], name),
                          "/leave_one_out.xls")
            if name in ["LiAlH4", "NaBH4", "MeLi","PhLi"]:
                df_test = dfs["test_" + name]
                df_test["mol"] = df_test["smiles"].apply(lambda smiles: df[df["smiles"] == smiles]["mol"].iloc[0])
                for mol, RT in zip(df_test["mol"], df_test["RT"]):
                    energy_to_Boltzmann_distribution(mol, RT)
                    feature_value(mol, dfp, param["feature"])
                prediction(model, df_test, dfp_[["r", "d", "t"]].values[dfp_["RMSE"].idxmin()], param["feature"],
                           "{}/{}".format(param["save_dir"], name), "/test_prediction.xls")
            print(dfp_)
            if False and name=="MeMgI":
                for name,n in zip(["EtMgI","iPrMgI","tBuMgI"],[1.1,1.2,1.3]):
                    df_test = dfs["test_" + name]
                    df_test["mol"] = df_test["smiles"].apply(lambda smiles: df[df["smiles"] == smiles]["mol"].iloc[0])

                    p=dfp_.iloc[dfp_["RMSE"].idxmin():dfp_["RMSE"].idxmin()+1]
                    p["r"]=p["r"]*n
                    #p["d"]=p["d"]*n
                    p=p.round(2)
                    print(p)
                    for mol, RT in zip(df_test["mol"], df_test["RT"]):
                        dirs_name = param["one_point_out_file"] + "/" + mol.GetProp("InchyKey")
                        pkl_to_featurevalue(dirs_name, p, mol, param["feature"])
                        energy_to_Boltzmann_distribution(mol, RT)
                        feature_value(mol, p, param["feature"])

                    prediction(model, df_test, p[["r", "d", "t"]].values[0], param["feature"],
                               "{}/{}".format(param["save_dir"], name), "/test_prediction.xls")