import itertools
import numpy as np

def addmesh(xyzview):
    radius=0.005
    rang=[[-5,2],[-5,5],[-4,4]]
    def addcylinder_func(rang,xyzview,radius):
        xyzview.addCylinder({"start": {"x": rang[0][0], "y": rang[1][0], "z": rang[2][0]},
                             "end": {"x": rang[0][1], "y": rang[1][1], "z": rang[2][1]},
                             "radius": radius, "color": "gray"})
        return xyzview
    if False:#内部のグリッド表示
        for x,y in itertools.product(range(rang[0][0]+1,rang[0][1]),range(rang[1][0]+1,rang[1][1])):
            xyzview=addcylinder_func([[x,x],[y,y],rang[2]],xyzview,radius)
        for y,z in itertools.product(range(rang[1][0]+1,rang[1][1]),range(rang[2][0]+1,rang[2][1])):
            xyzview = addcylinder_func([rang[0], [y, y], [z,z]], xyzview, radius)
        for z,x in itertools.product(range(rang[2][0]+1,rang[2][1]),range(rang[0][0]+1,rang[0][1])):
            xyzview = addcylinder_func([[x,x], rang[1], [z, z]], xyzview, radius)

    if True:#エッジのグリッド表示

        for x, y in itertools.product(rang[0],rang[1]):
            xyzview = addcylinder_func([[x, x], [y, y], rang[2]], xyzview, radius * 5)

        for y,z in itertools.product(rang[1], rang[2]):
            xyzview = addcylinder_func([rang[0], [y, y], [z,z]], xyzview, radius*5)

        for z,x in itertools.product(rang[2], rang[0]):
            xyzview = addcylinder_func([[x,x], rang[1], [z, z]], xyzview, radius*5)

    if True: #面のグリッド表示
        l=list(itertools.product(rang[0],range(rang[1][0],rang[1][1]+1)))+\
          list(itertools.product(range(rang[0][0],rang[0][1]+1),rang[1]))
        for x, y in l:
            xyzview = addcylinder_func([[x, x], [y, y], rang[2]], xyzview, radius * 2)

        l=list(itertools.product(rang[1],range(rang[2][0],rang[2][1]+1)))+\
          list(itertools.product(range(rang[1][0],rang[1][1]+1),rang[2]))
        for y,z in l:
            xyzview = addcylinder_func([rang[0], [y, y], [z,z]], xyzview, radius*2)

        l=list(itertools.product(rang[2],range(rang[0][0],rang[0][1]+1)))+\
          list(itertools.product(range(rang[2][0],rang[2][1]+1),rang[0]))
        for z,x in l:
            xyzview = addcylinder_func([[x,x], rang[1], [z, z]], xyzview, radius*2)



def addxyzarrow(xyzview):
    d=4
    radius=0.025
    color="gray"
    xyzview.addArrow(
        {"start": {"x": -d, "y": 0, "z": 0},
         "end": {"x": d, "y": 0, "z": 0}, "radius": radius,
         "radiusRatio": 4, "mid": 0.9, "color": color})
    xyzview.addArrow(
        {"start": {"x": 0, "y": -d, "z": 0},
         "end": {"x": 0, "y": d, "z": 0}, "radius": radius,
         "radiusRatio": 4, "mid": 0.9, "color": color})
    xyzview.addArrow(
        {"start": {"x": 0, "y": 0, "z": -d},
         "end": {"x": 0, "y": 0, "z": d}, "radius": radius,
         "radiusRatio": 4, "mid": 0.9, "color": color})
    xyzview.addLabel("x", {"position": {"x": d, "y": 0, "z": 0},
                           "backgroundColor": color, "backgroundOpacity": 0.5})
    xyzview.addLabel("y", {"position": {"x": 0, "y": d, "z": 0},
                           "backgroundColor": color, "backgroundOpacity": 0.5})
    xyzview.addLabel("z", {"position": {"x": 0, "y": 0, "z": d},
                           "backgroundColor": color, "backgroundOpacity": 0.5})

def add_label(xyz,param,center):
    radius=0.05
    radiusRatio=3
    color="black"
    xyz.addArrow({"start": {"x": 0, "y": 0, "z": 0},
                         "end": {"x": center[0], "y": 0, "z": center[2]},
                         "radius": radius, "color": color,"radiusRatio": radiusRatio,"mid": 0.8})
    xyz.addArrow({"start": {"x": 0, "y": 0, "z": 0},
                     "end": {"x": center[0], "y": 0, "z": -center[2]},
                     "radius": radius, "color": color,"radiusRatio": radiusRatio,"mid": 0.8})
    # xyz.addCylinder({"start": {"x": 0, "y": 0, "z": 0},
    #                  "end": {"x": param[1], "y": 0, "z": 0},
    #                  "radius": radius, "color": color})

    xyz.addArrow({"start": {"x": center[0]+param[0]/2,"y":center[1],"z":center[2]},
                  "end": {"x": center[0]+param[0],"y":center[1],"z":center[2]},
                  "radius": radius, "color": color, "radiusRatio": radiusRatio, "mid": 0.5})
    xyz.addArrow({"start": {"x": center[0]+param[0]/2, "y": center[1], "z": center[2]},
                  "end": {"x": center[0], "y": center[1], "z": center[2]},
                  "radius": radius, "color": color, "radiusRatio": radiusRatio, "mid": 0.5})
    xyz.addArrow({"start": {"x": center[0] + param[0] / 2, "y": center[1], "z": -center[2]},
                  "end": {"x": center[0] + param[0], "y": center[1], "z": -center[2]},
                  "radius": radius, "color": color, "radiusRatio": radiusRatio, "mid": 0.5})
    xyz.addArrow({"start": {"x": center[0] + param[0] / 2, "y": center[1], "z": -center[2]},
                  "end": {"x": center[0], "y": center[1], "z": -center[2]},
                  "radius": radius, "color": color, "radiusRatio": radiusRatio, "mid": 0.5})
    """
    xyz.addCurve({"points": [{"x":center[0],"y":center[1],"z":center[2]}, {"x":center[0]+param[0],"y":center[1],"z":center[2]}],
                              "radius":0.1,
                              "fromArrow":True,
                              "toArrow": True,
                              "color":'orange',
                              })
                              """
    for t in np.arange(0,param[2],1):
        xyz.addCylinder({"start": {"x": param[1]/2*np.cos(np.radians(t)), "y": 0, "z": param[1]/2*np.sin(np.radians(t))},
                             "end": {"x": param[1]/2*np.cos(np.radians(t+1.1)), "y": 0, "z": param[1]/2*np.sin(np.radians(t+1.1))},
                             "radius": radius, "color": color})
        xyz.addCylinder(
            {"start": {"x": param[1]/2 * np.cos(np.radians(t)), "y": 0, "z": -param[1]/2 * np.sin(np.radians(t))},
             "end": {"x": param[1]/2 * np.cos(np.radians(t + 1.1)), "y": 0, "z": -param[1]/2 * np.sin(np.radians(t + 1.1))},
             "radius": radius, "color": color})
    """xyz.addArrow({"start": {"x": param[1]*np.cos(np.radians(param[2]-10)), "y": 0, "z": param[1]*np.sin(np.radians(param[2]-10))},
                  "end": {"x": param[1]*np.cos(np.radians(param[2])), "y": 0, "z": param[1]*np.sin(np.radians(param[2]))},
                  "radius": radius, "color": "blue", "radiusRatio": 2, "mid": 0})"""

    """xyz.addLabel("{} [deg.]".format("θ"), {"position": {"x": param[1]*np.cos(np.radians(param[2]/2)), "y": 0, "z": param[1]*np.sin(np.radians(param[2]/2))},
                           "backgroundColor": "black", "backgroundOpacity": 0.5})
    xyz.addLabel("{} [deg.]".format("-θ"), {"position": {"x": param[1] * np.cos(np.radians(param[2] / 2)), "y": 0,
                                                        "z": -param[1] * np.sin(np.radians(param[2] / 2))},
                                           "backgroundColor": "black", "backgroundOpacity": 0.5})
    xyz.addLabel("{} [Å]".format("d"), {"position": {"x": param[1] * np.cos(np.radians(param[2]))/2, "y": 0,
                                                        "z": param[1] * np.sin(np.radians(param[2]))/2},
                                           "backgroundColor": "black", "backgroundOpacity": 0.5})
    xyz.addLabel("{} [Å]".format("d"), {"position": {"x": param[1] * np.cos(np.radians(param[2]))/2, "y": 0,
                                                         "z": -param[1] * np.sin(np.radians(param[2]))/2},
                                            "backgroundColor": "black", "backgroundOpacity": 0.5})
    xyz.addLabel("{} [Å]".format("r"), {"position": {"x": center[0]+param[0]/2,"y":center[1],"z":center[2]},
                                        "backgroundColor": "black", "backgroundOpacity": 0.5})"""
    xyz.addSphere(
        {"center": {"x": center[0], "y": center[1], "z": center[2]}, 'opacity': 1, "radius": radius * 2,
         "color": "black"})
    xyz.addSphere(
        {"center": {"x": center[0], "y": center[1], "z": -center[2]}, 'opacity': 1, "radius": radius*2,
         "color": "black"})