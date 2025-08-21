from openvino import Model
from sympy.codegen.ast import uint8
from torch.ao.nn.quantized.functional import threshold

from potato_tracker2 import Potato, Potato_tracker, MainProgram  # Import klasy i trakcera
import numpy as np
import cv2
from potato_tracker2 import Potato
from potato_tracker2 import PotatoCategorizer
from potato_tracker2 import ModelHandler


program = MainProgram(r"D:\Pycharm\Projekt3KMK\runs\segment\train4\weights\best_openvino_model\best.xml", "video_640.mp4", categorizer_threshold=10000, use_redis=False)
program.start_processing()





