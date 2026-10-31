import os
import re

staging_dir = r"C:\Users\LENOVO\Pictures\MinorProject\ActualImplementation_Kaggle"

# Common data helper
common_helper_code = '''import os
import scipy.io as sio
import numpy as np

def get_hsi_data_dir():
    candidates = [
        "/kaggle/input/dataset-mp",
        "/kaggle/input/minor-project-data",
        "/kaggle/input/datasets/tejassurya2004/dataset-mp",
        r"C:\\Users\\LENOVO\\Pictures\\MinorProject\\data",
        "../data",
        "../../data",
        "./data"
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.exists(os.path.join(c, "Indian_pines_corrected.mat")):
            return os.path.abspath(c)
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "Indian_pines_corrected.mat" in files:
                return root
    return "."

def load_indian_pines():
    data_dir = get_hsi_data_dir()
    img_path = os.path.join(data_dir, "Indian_pines_corrected.mat")
    gt_path = os.path.join(data_dir, "Indian_pines_gt.mat")
    img = sio.loadmat(img_path)["indian_pines_corrected"]
    gt = sio.loadmat(gt_path)["indian_pines_gt"]
    return img, gt
'''

with open(os.path.join(staging_dir, "common_data.py"), "w", encoding="utf-8") as f:
    f.write(common_helper_code)
print("Saved common_data.py")
