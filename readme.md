# sphere model is a program that analyzes three-dimensional information from cube files

## environment

Please build an environment based on the following files.
environment.yml

## calculation

First, a dataset is created from the hand labeling dataset to be used in the calculation.
If you run sphere_model_lib/dataset.py, hand labeling data is read and arranged_dataset was obtained.

Then, executing sphere_model_lib/calculate_conformation.py performs the structural optimization calculation, and executing sphere_model_lib/calculate_cubefile.py performs the cube information calculation.
Calculation methods will be set by parameter/optimization_parameter and parameter/run_sphere_model_parameter.

After the cube information has been computed, you can run sphere model.
Sphere parameter and calculation methods will be set by sphere_model_lib/sphere_parameter.py and parameter/run_sphere_model_parameter, respectively.
Calculation and regression are done by sphere_model_lib/sphere_parameter.py

Graph is drawn by sphere_model_lib/graph_all.py