# sphere model is a program that analyzes three-dimensional information from cube files

# 1. SETUP
## 1-1. git clone
Clone this code by following:

```
git clone -b {ブランチ名} https://github.com/poclab-web/sphere_model.git
```

Example:
```
git clone -b main https://github.com/poclab-web/sphere_model.git
```

## 1.2 environment built
Please build an environment based on the following files.
```
environment.yml
```
# 2.HOW TO USE
## 2.1 dataset
以下のファイルが元のデータセットです。
```
dataset/data.xlsx
```
元のデータセットから、以下のpyファイルで任意の条件で絞り込んでデータを作成します。
```
sphere_model_lib/dataset.py
```
上の方のコードに、入力するパスと出力するパスがあります。デフォルトでは以下のようになっています。
```
from_file_path = "../dataset/data.xlsx"
to_dir_path = "../arranged_dataset_1020"
```
38行目のコードによって構造の絞り込みを行っています。SMARTS形式で入力します。例えば、以下のコードではケトン基のベータ位にN, OH, F, Cl, Brが入るのを除きます。
```
not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]C(=O)[#6][N,OH1,F,Cl,Br]")
```
正しく計算が終われば、以下のパスに出力されます。
```
to_dir_path = "../arranged_dataset_1020"
```
## 2.2 Structural optimization
構造最適化計算は、以下のpyファイルを用いて計算します。
```
sphere_model_lib/calculate_conformation.py
```
計算条件は以下のパラメータファイルで指定します。
```
for param_file_name in glob.glob(
            "../parameter/optimization_parameter_1221/*.txt")
```
パラメータファイルを見てみると、以下のようになっています。ここでは、コメントアウトのあとにそれぞれの変数の意味を説明します。適宜設定してください。
```
{
"numConfs": 10000, #Distance geometry法で発生させる初期構造数
"cut_MMFF_energy":4,#MMFF計算を行った後、不安定はいざを排除するしきい値（再安定はいざからのエネルギー[kcal/mol]）
"cut_MMFF_rmsd":0.5, # MMFF計算を行った後、構造が類似のはいざを排除するしきい値（重原子のRMSD[Å]）
"max_MMFF_conformer":5,　#MMFF計算で最終的に許容する配座数
"MMFF_opt_save_file":"/Volumes/HD-PGF-A/MMFF_optimization", # MMFF計算の構造最適化ファイルの出力先

"optimize_level": "b3lyp/6-31g(d)",　#DFT計算での構造最適化レベル。初期構造は"MMFF_opt_save_file"で指定したパスを読み取ります。
"cut_psi4_energy":false, #DFT計算を行ったあと、不安定はいざを排除するしきい値（再安定はいざからのエネルギー[kcal/mol]）
"cut_psi4_rmsd":false,# DFT計算を行ったあと、構造が類似のはいざを排除するしきい値（重原子のRMSD[Å]）
"max_psi4_conformer":false,　#DFT計算で最終的に許容する配座数
"psi4_opt_save_file":"/Volumes/HD-PGF-A/b3lyp_6-31g(d)_optimization" #DFT計算での構造最適化ファイルの出力先
}
```

計算する分子は、パスで指定します。デフォルトでは以下のとおりです。この中に存在するsmilesカラムの情報をSMILES形式で読み取って、計算します。
```
datafilename = "../arranged_dataset_1020/*.xls"
```
## 2.3 Structural optimization
一点計算は、以下のpyファイルを用いて計算します。
```
sphere_model_lib/calculate_cubefile.py
```
計算条件は以下のパラメータファイルで指定します。デフォルトではfor文になっています。
```
for param_file_name in glob.glob("../parameter/single_point_calculation_parameter_1222/*.txt"):
```
パラメータファイルを見てみると、以下のようになっています。ここでは、コメントアウトのあとにそれぞれの変数の意味を説明します。適宜設定してください。
```
{
"psi4_aligned_save_file":"/Volumes/SSD-PSM960U3-UW/b3lyp_6-31g(d)_optimization",　＃計算する構造が入ったファイル名。
"one_point_level": "b3lyp/6-31g(d)",　＃計算レベル
"one_point_out_file":"/Volumes/SSD-PSM960U3-UW/b3lyp_6-31g(d)_b3lyp_6-31g(d)_single_point_calculation"　＃出力先のパス
}
```

計算する分子は、パスで指定します。デフォルトでは以下のとおりです。この中に存在するsmilesカラムの情報をSMILES形式で読み取って、計算します。
```
datafilename = "../arranged_dataset_1020/*.xls"
```
## 2.4 Calculation
以下のパイファイルを動かして球パラメータを指定します。
```
sphere_model_lib/sphere_parameter.py
```
デフォルトでは以下のようになっていますが、適宜調整してください。
```
delta = 0.2
r = np.arange(1, 4.1, delta)
t = np.arange(70, 121, 5)
l = []
for r_ in r:
    for d_ in np.arange(r_, r_ + 1.1, delta):
        for t_ in t:
            l.append([r_, d_, t_])
print(len(r), len(t), len(l))
dfp = pd.DataFrame(l, columns=['r', 'd', "t"])
dfp = dfp.assign(x=dfp["d"] * np.cos(np.radians(dfp["t"])),
                 y=0,
                 z=dfp["d"] * np.sin(np.radians(dfp["t"])))
dfp.to_csv("../sphere_parameter/light.csv")
```

以下のパイファイルを動かしてsphere model を実行します。
```
sphere_model_lib/run_sphere_model.py
```
実行パラメータは以下のパスで指定します。
```
for param_file_name in glob.glob("../parameter/run_sphere_model_parameter_1120/*.txt"):
```
パラメータの中身を説明します。
```
{
"one_point_out_file":"/Volumes/SSD-PSM960U3-UW/b3lyp_6-31g(d)_b3lyp_6-31g(d)_single_point_calculation",#一点計算の出力先
"data_file_name":"../arranged_dataset_1020/{}.xls",# データセットのパス
"feature":["Dt", "LUMO"],　＃解析する情報
"sphere_parameter_path": "../sphere_parameter/light.csv",# 球パラメータ
"save_dir": "../results/b3lyp_6-31g(d)_b3lyp_6-31g(d)_Dt_LUMO_1120",#結果の保存先
"fig_dir":"../figs/b3lyp_6-31g(d)_b3lyp_6-31g(d)_Dt_LUMO_1120",＃図の出力先（ここではまだ図は出ません）
"single point calculation":"b3lyp/6-31g(d)"　＃一点計算の名前（作図で使います。）
}
```
計算が無事に終了すれば、"save_dir"で指定したパスに結果が出力されます。

図は以下のパイファイルで作図します。
```
sphere_model_lib/graph_all.py
```
正しく計算できれば、"fig_dir"で指定したパスに図が出力されます。

# memo

First, a dataset is created from the hand labeling dataset to be used in the calculation.
If you run sphere_model_lib/dataset.py, hand labeling data is read and arranged_dataset was obtained.

Then, executing sphere_model_lib/calculate_conformation.py performs the structural optimization calculation, and executing sphere_model_lib/calculate_cubefile.py performs the cube information calculation.
Calculation methods will be set by parameter/optimization_parameter and parameter/run_sphere_model_parameter.

After the cube information has been computed, you can run sphere model.
Sphere parameter and calculation methods will be set by sphere_model_lib/sphere_parameter.py and parameter/run_sphere_model_parameter, respectively.
Calculation and regression are done by sphere_model_lib/sphere_parameter.py

Graph is drawn by sphere_model_lib/graph_all.py