import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_squared_error
import json
import glob
import numpy as np
from sklearn import linear_model

# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager#._rebuild()
plt.rcParams['font.family'] = 'Times New Roman'  # 全体のフォントを設定

os.makedirs("../figs/all", exist_ok=True)

if True:
    fig = plt.figure(figsize=(11, 3))
    j = 0
    for param_file_name, color in zip(sorted(glob.glob("../parameter/run_sphere_model_parameter_1120/*.txt")),
                                      ["red", "blue", "green"]):
        j += 1
        ax = fig.add_subplot(1, 3, j)

        with open(param_file_name, "r") as f:
            param = json.loads(f.read())
        print(param)
        os.makedirs(param["fig_dir"], exist_ok=True)
        coefs = []
        Rs = []
        for i, name in enumerate(["BH3", "LiAlH4", "NaBH4", "LiAl(OMe)3H", "MeLi", "MeMgI", "PhLi", "PhMgI"]):
            dfp = pd.read_csv("{}/{}/grid_search.csv".format(param["save_dir"], name))
            coef = dfp[["b_electron", "b_LUMO"]].values[dfp["RMSE"].idxmin()]
            coef = -coef[1] * 100 / np.sum(np.abs(coef))
            coefs.append(coef)
            R = dfp[["r"]].values[dfp["RMSE"].idxmin()]
            Rs.append(R)
            print(coef, R, name)
        cmap = plt.cm.get_cmap("cool_r")
        sc = ax.scatter(Rs, coefs, c=coefs, cmap=cmap, s=np.array(Rs) * 40)
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.0)

        plt.colorbar(sc, cax=cax, ticks=[0, 20, 40, 60])
        # ax.plot(coefs, Rs, "o", color=color)  # ,color="black")
    # ax.set_xlim([0, 8])
    # ax.set_ylim([0, -1])
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylabel("orbital contribution [%]")
    ax.set_xlabel("optimam radius [Å]")
    fig.tight_layout()  # レイアウトの設定
    plt.savefig("../figs/all/coef_R.png", dpi=500)

if True:
    fig = plt.figure(figsize=(14, 6))
    for i, (nu, label) in enumerate(zip(["BH3", "LiAlH4", "NaBH4", "LiAl(OMe)3H", "MeLi", "MeMgI", "PhLi", "PhMgI"],
                                        ["{BH_3}", "{LiAlH_4}", "{NaBH_4}", "{LiAl(OMe)_3H}", "{MeLi}", "{MeMgI}",
                                         "{PhLi}",
                                         "{PhMgI}"])):
        ax = fig.add_subplot(2, 4, i + 1)
        for param_file_name, color in zip(sorted(glob.glob("../parameter/run_sphere_model_parameter_1120/*.txt")),
                                          [["red", "red"], ["blue", "blue"], ["green", "green"]]):

            with open(param_file_name, "r") as f:
                param = json.loads(f.read())
            print(param)
            level = param["single point calculation"] + "//B3LYP/6-31G(d)"
            df = pd.read_excel("{}/{}/leave_one_out.xls".format(param["save_dir"], nu))

            ax.plot([-2.5, 2.5], [-2.5, 2.5], color="Gray", alpha=0.5)
            if i == 7:
                ax.plot(df["ΔΔG.expt."], df["ΔΔG.regression"], "x", color=color[0], markersize=4,
                        alpha=0.2,
                        label=level + " regression")  # "regression $r^2$ = {:.2f}".format(r2_score(df["ΔΔG.expt."], df["ΔΔG.train"]))
                ax.plot(df["ΔΔG.expt."], df["ΔΔG.loo"], "s", color='none', markersize=4, alpha=0.5,
                        markeredgecolor=color[1],
                        label=level + " LOOCV")  # , label="leave-one-out "+"$q^2 = {:.2f}$".format(r2_score(df["ΔΔG.expt."], df["ΔΔG.loo"]))
                ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1., 0.5))
            else:
                ax.plot(df["ΔΔG.expt."], df["ΔΔG.regression"], "x", color=color[0], markersize=4,
                        alpha=0.2)  # , label="regression $r^2$ = {:.2f}".format(r2_score(df["ΔΔG.expt."], df["ΔΔG.train"]))
                ax.plot(df["ΔΔG.expt."], df["ΔΔG.loo"], "s", color='none', markersize=4, alpha=0.5,
                        markeredgecolor=color[
                            1])  # , label="leave-one-out "+"$q^2 = {:.2f}$".format(r2_score(df["ΔΔG.expt."], df["ΔΔG.loo"]))

            ax.set_xticks([-2.5, 0, 2.5])
            ax.set_yticks([-2.5, 0, 2.5])
            # ax.set_xlabel("experimental ΔΔG [kcal/mol]", fontsize=10)#"$\mathrm{ΔΔG_{expt.}}$ [kcal/mol]"
            # ax.set_ylabel("predicted ΔΔG [kcal/mol]", fontsize=10)#"$\mathrm{ΔΔG_{pred.}}$ [kcal/mol]"
            ax.set_xlabel("ΔΔ${G_{expt}}$ [kcal/mol]", fontsize=10)
            ax.set_ylabel("ΔΔ${G_{predict}}$ [kcal/mol]", fontsize=10)
            ax.set_title("$\mathrm{}$ training dataset".format(label), fontsize=10)
            print("counts = {} r2= {:.2f}, RMSE = {:.2f}, q2 = {:.2f}, RMSE = {:.2f}, {}"
                  .format(len(df),
                          r2_score(df["ΔΔG.expt."], df["ΔΔG.regression"]),
                          np.sqrt(mean_squared_error(df["ΔΔG.regression"], df["ΔΔG.expt."])),
                          r2_score(df["ΔΔG.expt."], df["ΔΔG.loo"]),
                          np.sqrt(mean_squared_error(df["ΔΔG.loo"], df["ΔΔG.expt."]))
                          , label))
        fig.tight_layout()
    plt.savefig("../figs/all/prediction_training.png", dpi=500)

fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(1, 1, 1)

fig = plt.figure(figsize=(14, 3))
for i, (name, label) in enumerate(
        zip(["LiAlH4", "NaBH4", "MeLi", "PhLi"], ["{LiAlH_4}", "{NaBH_4}", "{MeLi}", "{PhLi}"])):
    ax = fig.add_subplot(1, 4, i + 1)
    for param_file_name, color in zip(sorted(glob.glob("../parameter/run_sphere_model_parameter_1120/*.txt")),
                                      [["red", "red"], ["blue", "blue"], ["green", "green"]]):
        with open(param_file_name, "r") as f:
            param = json.loads(f.read())
        print(param)
        level = param["single point calculation"] + "//B3LYP/6-31G(d)"
        df_test = pd.read_excel("{}/{}/test_prediction.xls".format(param["save_dir"], name))
        if name == "PhLi":
            df_test = df_test  # .drop([5, 6, 7, 8])
        df_train = pd.read_excel("{}/{}/leave_one_out.xls".format(param["save_dir"], name))
        ax.plot([-2.5, 2.5], [-2.5, 2.5], color="Gray", alpha=0.5)

        if i == 3:
            df_test = df_test  # [:5]
            ax.plot(df_train["ΔΔG.expt."], df_train["ΔΔG.regression"], "x", color=color[0], markersize=4,
                    alpha=0.2,
                    label=level + " training")  # , label="train $r^2$ = {:.2f}".format(r2_score(df_train["ΔΔG.expt."], df_train["ΔΔG.train"]))
            ax.plot(df_test["ΔΔG.expt."], df_test["ΔΔG.predict"], "o", color="none", markeredgecolor=color[1],
                    markersize=4,
                    alpha=0.5,
                    label=level + " test")  # label="test $r^2_{test}$ = "+"{:.2f}".format(r2_score(df_test["ΔΔG.expt."], df_test["ΔΔG.predict"])),
            ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1., 0.5))
        else:
            ax.plot(df_train["ΔΔG.expt."], df_train["ΔΔG.regression"], "x", color=color[0], markersize=4,
                    alpha=0.2)  # , label="train $r^2$ = {:.2f}".format(r2_score(df_train["ΔΔG.expt."], df_train["ΔΔG.train"]))
            ax.plot(df_test["ΔΔG.expt."], df_test["ΔΔG.predict"], "o", color="none", markeredgecolor=color[1],
                    markersize=4,
                    alpha=0.5)  # label="test $r^2_{test}$ = "+"{:.2f}".format(r2_score(df_test["ΔΔG.expt."], df_test["ΔΔG.predict"])),
        print(
            "{} r2= {:.2f}, r2_test = {:.2f}, R2 = {:.2f}, k = {:.2f}, r2'= {:.2f}, r2_test' = {:.2f}, RMSE = {:.2f}, {} test_train"
            .format(len(df_test),
                    r2_score(df_train["ΔΔG.expt."], df_train["ΔΔG.regression"]),
                    r2_score(df_test["ΔΔG.expt."], df_test["ΔΔG.predict"]),
                    df_test[["ΔΔG.expt.", "ΔΔG.predict"]].corr().iloc[0][1],
                    linear_model.LinearRegression(fit_intercept=False).fit(
                        df_test["ΔΔG.predict"].values.reshape((-1, 1)),
                        df_test["ΔΔG.expt."]).coef_[0],
                    r2_score(df_train["ΔΔG.regression"], df_train["ΔΔG.expt."]),
                    r2_score(df_test["ΔΔG.predict"], df_test["ΔΔG.expt."]),
                    np.sqrt(mean_squared_error(df_test["ΔΔG.predict"], df_test["ΔΔG.expt."])), name))

    # ax.legend(fontsize=6,loc='lower right')  # 凡例
    ax.set_xticks([-2.5, 0, 2.5])
    ax.set_yticks([-2.5, 0, 2.5])
    # ax.set_xlabel("experimental ΔΔG [kcal/mol]", fontsize=10)#"$\mathrm{ΔΔG_{expt.}}$ [kcal/mol]"
    # ax.set_ylabel("predicted ΔΔG [kcal/mol]", fontsize=10)#"$\mathrm{ΔΔG_{pred.}}$ [kcal/mol]"
    ax.set_xlabel("ΔΔ${G_{expt}}$ [kcal/mol]", fontsize=10)
    ax.set_ylabel("ΔΔ${G_{predict}}$ [kcal/mol]", fontsize=10)
    ax.set_title("$\mathrm{}$".format(label) + " training / test dataset"
                 , fontsize=10)
# ax.legend([None,"$\mathrm{b_{LUMO}}$","$\mathrm{b_{electron}}$"], loc='upper left', bbox_to_anchor=(0.5, 1.1),  ncol=2)
fig.tight_layout()
plt.savefig("../figs/all/prediction_test.png", dpi=500)
for param_file_name, color in zip(sorted(glob.glob("../parameter/run_sphere_model_parameter_1120/*.txt")),
                                  ["red", "blue", "green"]):
    with open(param_file_name, "r") as f:
        param = json.loads(f.read())
    os.makedirs(param["fig_dir"], exist_ok=True)
    with open(param["fig_dir"] + "/summery.txt", "w") as f:
        for i, name in enumerate(["BH3", "LiAlH4", "NaBH4", "LiAl(OMe)3H", "MeLi", "MeMgI", "PhLi", "PhMgI"]):
            dfp = pd.read_csv("{}/{}/grid_search.csv".format(param["save_dir"], name))
            # coef = dfp[["Dt_coef", "LUMO_coef"]].values[dfp["RMSE"].idxmin()]
            print(name, file=f)
            print(dfp.drop(['x', 'y', 'z'], axis='columns').iloc[dfp["RMSE"].idxmin()][
                      ["r", "d", "t", "b_electron", "b_LUMO"]].round(2), file=f)

# feature importance
fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(1, 1, 1)
for param_file_name, color in zip(sorted(glob.glob("../parameter/run_sphere_model_parameter_1120/*.txt")),
                                  ["red", "blue", "green"]):
    with open(param_file_name, "r") as f:
        param = json.loads(f.read())
    print(param)
    os.makedirs(param["fig_dir"], exist_ok=True)
    for i, name in enumerate(["BH3", "LiAlH4", "NaBH4", "LiAl(OMe)3H", "MeLi", "MeMgI", "PhLi", "PhMgI"]):
        dfp = pd.read_csv("{}/{}/grid_search.csv".format(param["save_dir"], name))
        coef = dfp[["b_electron", "b_LUMO"]].values[dfp["RMSE"].idxmin()]
        print(coef)
        ax.plot(coef[0:1], coef[1:2], "o", color=color)  # ,color="black")
ax.set_xlim([0, 8])
ax.set_ylim([0, -1])
ax.set_xticks([0, 4, 8])
ax.set_yticks([0, -0.5, -1])
ax.set_xlabel("${b_\mathrm{electron}}$ [kcal/mol]")
ax.set_ylabel("${b_\mathrm{LUMO}}$ [kcal/mol]")
fig.tight_layout()  # レイアウトの設定
plt.savefig("../figs/all/coef.png", dpi=500)

for param_file_name, color in zip(sorted(glob.glob("../parameter/run_sphere_model_parameter_1120/*.txt")),
                                  ["red", "blue", "green"]):

    fig, ax = plt.subplots()
    with open(param_file_name, "r") as f:
        param = json.loads(f.read())
    print(param)
    os.makedirs(param["fig_dir"], exist_ok=True)
    labels = ["{BH_3}", "{LiAlH_4}", "{NaBH_4}", "{LiAl(OMe)_3H}", "{MeLi}", "{MeMgI}",
              "{PhLi}",
              "{PhMgI}"]
    for i, (name, label) in enumerate(zip(["BH3", "LiAlH4", "NaBH4", "LiAl(OMe)3H", "MeLi", "MeMgI", "PhLi", "PhMgI"],
                                          labels)):
        dfp = pd.read_csv("{}/{}/grid_search.csv".format(param["save_dir"], name))
        coef = dfp[["b_electron", "b_LUMO"]].values[dfp["RMSE"].idxmin()]
        coef = np.abs(coef)
        coef = coef / np.sum(coef) * 100
        print(coef)
        ax.bar("$\mathrm{}$".format(label), coef[1], bottom=coef[0], color="blue")  # ,label="orbital"
        ax.bar("$\mathrm{}$".format(label), coef[0], color="orange")  # ,label="steric"
    ax.set_xticklabels(["$\mathrm{}$".format(label) for label in labels], rotation=20)
    ax.set_ylim([0, 100])

    ax.set_xlabel("training dataset")
    ax.set_ylabel("absolute value ratio %")

    ax.legend(["$\mathrm{b_{LUMO}}$", "$\mathrm{b_{electron}}$"], loc='upper left', bbox_to_anchor=(0.5, 1.1), ncol=2)
    fig.tight_layout()  # レイアウトの設定
    plt.savefig("../figs/all/coef_bar{}.png".format(color), dpi=500)
###
# 棒の配置位置、ラベルを用意

labels = ["{BH_3}", "{LiAlH_4}", "{NaBH_4}", "{LiAl(OMe)_3H}", "{MeLi}", "{MeMgI}",
          "{PhLi}",
          "{PhMgI}"]

# マージンを設定
margin = 0.2  # 0 <margin< 1
totoal_width = 1 - margin
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
fig, ax = plt.subplots(figsize=(6.4, 4))
levels = []
for i, (param_file_name, color) in enumerate(
        zip(sorted(glob.glob("../parameter/run_sphere_model_parameter_1120/*.txt")),
            ["red", "blue", "green"])):
    with open(param_file_name, "r") as f:
        param = json.loads(f.read())
    print(param)
    levels.append(param["single point calculation"])
    l = []
    for j, (name, label) in enumerate(zip(["BH3", "LiAlH4", "NaBH4", "LiAl(OMe)3H", "MeLi", "MeMgI", "PhLi", "PhMgI"],
                                          labels)):
        dfp = pd.read_csv("{}/{}/grid_search.csv".format(param["save_dir"], name))
        coef = dfp[["b_electron", "b_LUMO"]].values[dfp["RMSE"].idxmin()]
        # coef_ = coef / np.sum(np.abs(coef)) * 100
        l.append(coef.tolist())
    l = np.array(l)
    pos = x - totoal_width * (1 - (2 * i + 1) / 3) / 2
    # plt.bar( h, width=totoal_width / len(data))
    bar1 = ax.bar(pos, l[:, 1] / np.sum(np.abs(l), axis=1), width=totoal_width / 3, bottom=1, color="blue",
                  alpha=0.2 + 0.2 * np.sqrt(i))  # ,label="orbital"
    bar2 = ax.bar(pos, l[:, 0] / np.sum(np.abs(l), axis=1), width=totoal_width / 3, color="red",
                  alpha=0.2 + 0.2 * np.sqrt(i))  # ,label="steric"
    ax.bar_label(bar1, labels=["{:.2f}".format(_) for _ in l[:, 1]], rotation=90, label_type='center',
                 fontsize=12)  # ,fmt='%.2f'
    ax.bar_label(bar2, labels=["+{:.2f}".format(_) for _ in l[:, 0]], rotation=90, label_type='center',
                 fontsize=12)  # ,fmt='%.2f'
ax.set_xticks(x)
ax.set_xlim(1 - (1 - margin) / 3 - margin, 8 + (1 - margin) / 3 + margin)
ax.set_ylim(0, 1.1)
ax.set_yticks([0, 0.5, 1], fontsize=12)
ax.set_xticklabels(["$\mathrm{}$".format(label) for label in labels], rotation=20)
ax.set_xlabel("training dataset", fontsize=12)
ax.set_ylabel("ratio", fontsize=12)

ax.legend(["${b_\mathrm{LUMO}}$    " + levels[0], "${b_\mathrm{electron}}$ " + levels[0],
           "${b_\mathrm{LUMO}}$    " + levels[1], "${b_\mathrm{electron}}$ " + levels[1],
           "${b_\mathrm{LUMO}}$    " + levels[2], "${b_\mathrm{electron}}$ " + levels[2]],
          loc='upper left', bbox_to_anchor=(-0.05, 1.2), ncol=3, fontsize=8)
#ax.plot([1 - (1 - margin) / 3 - margin, 8 + (1 - margin) / 3 + margin], [0.5, 0.5], color="Gray", alpha=0.5)
fig.tight_layout()  # レイアウトの設定

plt.savefig("../figs/all/coef_bars.png", dpi=500)

# all
markersize = 3
alpha = 0.4
if True:
    fig = plt.figure(figsize=(9, 3))
    for j, (param_file_name, color) in enumerate(
            zip(sorted(glob.glob("../parameter/run_sphere_model_parameter_1120/*.txt")),
                ["red", "blue", "green"])):
        with open(param_file_name, "r") as f:
            param = json.loads(f.read())
        print(param)

        ax = fig.add_subplot(1, 3, 1 + j)
        l = []
        for i, (name, label) in enumerate(
                zip(["BH3", "LiAlH4", "NaBH4", "PhLi", "LiAl(OMe)3H", "MeLi", "MeMgI", "PhMgI"],
                    ["{BH_3}", "{LiAlH_4}", "{NaBH_4}", "{PhLi}", "{LiAl(OMe)_3H}", "{MeLi}",
                     "{MeMgI}",
                     "{PhMgI}"])):
            df = pd.read_excel("{}/{}/leave_one_out.xls".format(param["save_dir"], name))
            l.append(df)
        df_train = pd.concat(l)
        print(len(df_train), len(df_train.drop_duplicates(subset=['InchyKey'])))
        ax.plot([-2.5, 2.5], [-2.5, 2.5], color="Gray", alpha=alpha)
        if True:
            label = "regression $r^2$ = {:.2f}\nRMSE = {:.2f} kcal/mol".format(
                r2_score(df_train["ΔΔG.expt."], df_train["ΔΔG.regression"]),
                np.sqrt(mean_squared_error(df_train["ΔΔG.regression"], df_train["ΔΔG.expt."])))
        else:
            label = "training"
        ax.plot(df_train["ΔΔG.expt."], df_train["ΔΔG.regression"], "x", color="black",
                label=label,
                markersize=markersize, alpha=alpha)

        if True:  # _=="loo":
            label = "LOOCV " + "$q^2$" + " = {:.2f}\nRMSE = {:.2f} kcal/mol".format(
                r2_score(df_train["ΔΔG.expt."], df_train["ΔΔG.loo"]),
                np.sqrt(mean_squared_error(df_train["ΔΔG.loo"], df_train["ΔΔG.expt."])))
        else:
            label = "LOOCV"
        ax.plot(df_train["ΔΔG.expt."], df_train["ΔΔG.loo"], "o", color="blue",
                label=label,
                markersize=markersize, alpha=alpha)
        ax.set_xticks([-2.5, 0, 2.5])
        ax.set_yticks([-2.5, 0, 2.5])
        ax.set_xlabel("ΔΔ${G_{expt}}$ [kcal/mol]", fontsize=12)
        ax.set_ylabel("ΔΔ${G_{predict}}$ [kcal/mol]", fontsize=12)
        l = []
        for i, (name, label) in enumerate(
                zip(["LiAlH4", "NaBH4", "MeLi", "PhLi"], ["{LiAlH4}", "{NaBH_4}", "{MeLi}", "{PhLi}"])):
            df_test = pd.read_excel("{}/{}/test_prediction.xls".format(param["save_dir"], name))
            if name == "PhLi":
                df_test = df_test  # .drop([5,6,7,8])##[:5,10:]
            l.append(df_test)
        df_test = pd.concat(l)
        print(len(df_test), len(df_test.drop_duplicates(subset=['InchyKey'])))
        if True:  # _=="test":
            label = "test " + "${r_\mathrm{test}^2}$" + " = {:.2f}\nRMSE = {:.2f} kcal/mol".format(
                r2_score(df_test["ΔΔG.expt."], df_test["ΔΔG.predict"]),
                np.sqrt(mean_squared_error(df_test["ΔΔG.predict"], df_test["ΔΔG.expt."])))
        else:
            label = "test"
        ax.plot(df_test["ΔΔG.expt."], df_test["ΔΔG.predict"], "s", color="red",
                label=label,
                markersize=markersize, alpha=alpha)
        ax.text(-2.5, 2.0,
                "$\mathrm{N_{training}}$ = " + "{}\n".format(len(df_train)) + "$\mathrm{N_{test}}$ = " + "{}".format(
                    len(df_test)), fontsize=8)
        fig.tight_layout()  # レイアウトの設定
        ax.legend(fontsize=5.5, loc='lower right')  # 凡例
        ax.set_title(param["single point calculation"] + "//B3LYP/6-31G(d)", fontsize=10)
        print(
            "{} r2= {:.2f}, q2 = {:.2f}, r2_test = {:.2f}, R2 = {:.2f}, k = {:.2f}, r2'= {:.2f}, r2_test' = {:.2f}, "
            "RMSE_test = {:.2f}, RMSE_regression = {:.2f}, RMSE_LOOCV = {:.2f}, test_train"
            .format(len(df_test),
                    r2_score(df_train["ΔΔG.expt."], df_train["ΔΔG.regression"]),
                    r2_score(df_train["ΔΔG.expt."], df_train["ΔΔG.loo"]),
                    r2_score(df_test["ΔΔG.expt."], df_test["ΔΔG.predict"]),
                    df_test[["ΔΔG.expt.", "ΔΔG.predict"]].corr().iloc[0][1],
                    linear_model.LinearRegression(fit_intercept=False).fit(
                        df_test["ΔΔG.predict"].values.reshape((-1, 1)),
                        df_test["ΔΔG.expt."]).coef_[0],
                    r2_score(df_train["ΔΔG.regression"], df_train["ΔΔG.expt."]),
                    r2_score(df_test["ΔΔG.predict"], df_test["ΔΔG.expt."]),
                    np.sqrt(mean_squared_error(df_test["ΔΔG.predict"], df_test["ΔΔG.expt."])),
                    np.sqrt(mean_squared_error(df_train["ΔΔG.regression"], df_train["ΔΔG.expt."])),
                    np.sqrt(mean_squared_error(df_train["ΔΔG.loo"], df_train["ΔΔG.expt."]))))
    plt.savefig("../figs/all/prediction_all.png", dpi=500)
