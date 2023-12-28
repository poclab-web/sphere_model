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
dfp = pd.DataFrame(l, columns=['r', 'd', "t"])
dfp = dfp.assign(x=dfp["d"] * np.cos(np.radians(dfp["t"])),
                 y=0,
                 z=dfp["d"] * np.sin(np.radians(dfp["t"])))
dfp.to_csv("../sphere_parameter/light.csv")
