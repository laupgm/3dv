'''
from roboflow import Roboflow
rf = Roboflow(api_key="l0Nhx8ojiDzUae3Tdake")
project = rf.workspace("sugar-o862x").project("3dv")
version = project.version(1)
dataset = version.download("coco-segmentation")
'''

from roboflow import Roboflow
rf = Roboflow(api_key="l0Nhx8ojiDzUae3Tdake")
project = rf.workspace("sugar-o862x").project("3dv")
version = project.version(2)
dataset = version.download("yolov11")

'''
from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="3dv-1/test", use_segments=True)
'''