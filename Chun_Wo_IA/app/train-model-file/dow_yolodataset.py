from roboflow import Roboflow
rf = Roboflow(api_key="R7Lt6Qob58kGRwGwbMCX")
project = rf.workspace("work-place-6wdsa").project("stone-dirt")
version = project.version(5)
dataset = version.download("yolov8")
