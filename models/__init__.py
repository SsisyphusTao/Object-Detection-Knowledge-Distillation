from .CenternetMobilenetv3 import get_pose_net as init_student
from .experimental import attempt_load
import torch

def init_teacher():
    # Load model
    with torch.no_grad():
        model = attempt_load('./weights/helmet_head_person_l.pt') # load FP32 model
    model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names
    print('loading teacher with classes: ', names)
    return model