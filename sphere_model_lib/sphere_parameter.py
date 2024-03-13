import numpy as np
import pandas as pd

delta = 0.2
r = np.arange(1, 4.1, delta)
t = np.arange(70, 121, 5)
l = []
for r_ in r:
    for d_ in np.arange(r_, r_ + 1.1, delta):
        for t_ in t:
            l.append([r_, d_, t_])
print(len(r), len(t), len(l))
dfp = pd.DataFrame(l, columns=['R', 'd', "θ"])

dfp = dfp.assign(x=dfp["d"] * np.cos(np.radians(dfp["θ"])),
                 y=0,
                 z=dfp["d"] * np.sin(np.radians(dfp["θ"])))
dfp = dfp.round(decimals={
    'R': 1, 'd': 1, 'θ': 0,
    "x": 5, "y": 5, "z": 5})
print(dfp)
dfp.to_csv("../sphere_parameter/sphere_parameter.csv", index=False)
