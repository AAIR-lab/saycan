import numpy as np
import matplotlib
from matplotlib import colors

from pydantic import BaseModel
from typing import Tuple, List, Dict

class BodyInfo(BaseModel):

  position: Tuple[float, ...]
  orientation: Tuple[float, ...]

class State(BaseModel):

  index: int
  body_infos: Dict[str, BodyInfo]

#@markdown Global constants: pick and place objects, colors, workspace bounds

PICK_TARGETS = {
  "blue block": None,
  "red block": None,
  "green block": None,
  "yellow block": None,
}

COLORS = {
    "blue":   (78/255,  121/255, 167/255, 255/255),
    "red":    (255/255,  87/255,  89/255, 255/255),
    "green":  (89/255,  169/255,  79/255, 255/255),
    "yellow": (237/255, 201/255,  72/255, 255/255),
    "orange": colors.to_rgba(colors.TABLEAU_COLORS["tab:orange"]),
    "purple": colors.to_rgba(colors.TABLEAU_COLORS["tab:purple"]),
    "pink": colors.to_rgba(colors.TABLEAU_COLORS["tab:pink"]),
    "cyan": colors.to_rgba(colors.TABLEAU_COLORS["tab:cyan"]),
    "gray": colors.to_rgba(colors.TABLEAU_COLORS["tab:gray"]),
}

def get_objects(colors, object_type):

  assert object_type == "block" or object_type == "bowl"
  assert all([c in COLORS for c in colors])
  return ["%s %s" % (c, object_type) for c in colors]

PLACE_TARGETS = {
  "blue block": None,
  "red block": None,
  "green block": None,
  "yellow block": None,

  "blue bowl": None,
  "red bowl": None,
  "green bowl": None,
  "yellow bowl": None,

  "top left corner":     (-0.3 + 0.05, -0.2 - 0.05, 0),
  "top right corner":    (0.3 - 0.05,  -0.2 - 0.05, 0),
  "middle":              (0,           -0.5,        0),
  "bottom left corner":  (-0.3 + 0.05, -0.8 + 0.05, 0),
  "bottom right corner": (0.3 - 0.05,  -0.8 + 0.05, 0),
}


PIXEL_SIZE = 0.00267857
BOUNDS = np.float32([[-0.3, 0.3], [-0.8, -0.2], [0, 0.15]])  # X Y Z


#----------------------------------------VILD------------------------------------------
available_blocks = get_objects(COLORS.keys(), "block")
available_bowls = get_objects(COLORS.keys(), "bowl")

category_names = available_blocks + available_bowls
image_path = 'tmp.jpg'

#@markdown ViLD settings.
category_name_string = ";".join(category_names)
max_boxes_to_draw = 8 #@param {type:"integer"}

# Extra prompt engineering: swap A with B for every (A, B) in list.
prompt_swaps = [('block', 'cube')]

nms_threshold = 0.4 #@param {type:"slider", min:0, max:0.9, step:0.05}
min_rpn_score_thresh = 0.4  #@param {type:"slider", min:0, max:1, step:0.01}
min_box_area = 10 #@param {type:"slider", min:0, max:10000, step:1.0}
max_box_area = 3000  #@param {type:"slider", min:0, max:10000, step:1.0}
vild_params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area, max_box_area


# Coordinate map (i.e. position encoding).
coord_x, coord_y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224), sparse=False, indexing='ij')
coords = np.concatenate((coord_x[..., None], coord_y[..., None]), axis=2)