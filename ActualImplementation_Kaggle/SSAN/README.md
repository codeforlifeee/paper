# Spectral-Spatial Attention Network for Hyperspectral Image Classification

![](./figure.jpg)

## Paper

[Spectral-Spatial Attention Networks for Hyperspectral Image Classification](https://www.mdpi.com/2072-4292/11/8/963)

Please cite our paper if you find it useful for your research.

```
@Article{ssan,
  AUTHOR = {Mei, Xiaoguang and Pan, Erting and Ma, Yong and Dai, Xiaobing and Huang, Jun  and Fan, Fan and Du, Qinglei and Zheng, Hong and Ma, Jiayi},
  TITLE = {Spectral-Spatial Attention Networks for Hyperspectral Image Classification},
  JOURNAL = {Remote Sensing},
  VOLUME = {11},
  YEAR = {2019},
  NUMBER = {8},
  ARTICLE-NUMBER = {963},
  URL = {http://www.mdpi.com/2072-4292/11/8/963},
  ISSN = {2072-4292},
  DOI = {10.3390/rs11080963}
}
```

---

# 🚀 Extensive Guide: Running SSAN on Any Laptop

This codebase is currently configured to run the **Indian Pines** dataset (16 classes, 200 spectral bands). Follow these extensive, step-by-step instructions to set up the environment and train the model on *any* laptop.

## 1. Environment Setup

Deep learning models like this one require heavy computation. If your laptop has an **NVIDIA GPU** (like an RTX 3050, 4060, etc.) and runs Windows, it is highly recommended to use **WSL2** (Windows Subsystem for Linux) to automatically utilize your GPU.

If you are on a Mac or a laptop without a dedicated GPU, you can still run it! TensorFlow will automatically fallback to your CPU (though it will train slightly slower).

### Step 1.1: Prepare your OS
- **Windows (GPU / CPU):** Open PowerShell, type `wsl`, and hit enter. This puts you in a Linux environment natively linked to your Windows filesystem.
- **Mac / Linux (CPU / GPU):** Simply open your Terminal.

Navigate to the project directory. Example for Windows WSL:
```bash
# Replace this path with your exact folder location
cd /mnt/c/Users/YourName/Pictures/MinorProject/ActualImplementation/SSAN
```

### Step 1.2: Install Prerequisites
Ensure you have `python3` and `venv` installed.
If you are on Ubuntu/WSL, run:
```bash
sudo apt update
sudo apt install python3 python3-pip python3.12-venv
```
*(Note: Replace `3.12` with whatever Python version your system defaults to, e.g., `3.10`)*

### Step 1.3: Create and Activate a Virtual Environment
Virtual environments prevent conflicts between different Python projects.
```bash
# 1. Clear any broken environments (optional)
rm -rf venv 

# 2. Create the environment
python3 -m venv venv

# 3. Activate the environment
source venv/bin/activate
```
*(You should see `(venv)` appear at the left of your terminal prompt).*

### Step 1.4: Install Dependencies
With the environment active, install the required libraries. Modern TensorFlow versions have excellent backward compatibility for `.compat.v1` code, and will automatically utilize NVIDIA GPUs if detected.
```bash
pip install tensorflow numpy scipy scikit-learn matplotlib
```

---

## 2. Dataset Preparation

Ensure your dataset (`.mat` files) is placed correctly. By default, the code expects the data folder to be exactly one directory above the project folder, structured like this:
```
MinorProject/
│
├── data/
│   ├── Indian_pines_corrected.mat
│   └── Indian_pines_gt.mat
│
└── ActualImplementation/
    └── SSAN/
        ├── indices.py
        ├── save_indices.py
        └── ...
```

---

## 3. Running the Pipeline

Once your environment is set up and activated, you are ready to process the data and train the model.

### Step 3.1: Split the Dataset
The first script reads the `.mat` dataset, handles shuffling, and splits the pixels into `train`, `valid`, and `test` batches. It will output `.npy` index arrays into your folder.
```bash
python save_indices.py
```

### Step 3.2: Train the Model
This step initiates the Attention Network training loop. It will run for `3000` steps, outputting the loss and accuracy to the terminal periodically. Checkpoints (`.ckpt`) will be dynamically saved in the `model/` folder.
```bash
python ssan.py
```

### Step 3.3: Test the Model and Generate Maps
After the model finishes training, you want to evaluate its accuracy across the entire test set. 
```bash
python ssan_test.py
```
**Outputs of this script:**
1. It prints out the class-by-class testing accuracy to the terminal.
2. It generates **`merge.jpg`** (the model's predicted classification map) and **`gt.jpg`** (the ground truth map) directly in your project folder so you can visually compare the results.

---

## FAQ / Troubleshooting

**Q: I want to switch back to the Pavia University dataset.**
A: You will need to revert the hardcoded changes in the code:
1. In `save_indices.py`, change `Indian_pines_gt.mat` to `PaviaU_gt.mat` and `indian_pines_gt` to `paviaU_gt`.
2. In `indices.py`, change `input_dimension` back to `103`, and `num_classes` back to `9`. Swap the `.mat` file names and keys back to `paviaU`.
3. In `ssan.py` and `ssan_test.py`, change `num_classes` to `9` and `timesteps` to `103`.

**Q: I'm getting `ModuleNotFoundError: No module named 'tensorflow'`**
A: Ensure you have activated your virtual environment using `source venv/bin/activate` *before* running the python scripts.

**Q: My training is taking very long.**
A: If it's running on CPU, this is normal. To speed it up, ensure you are running it inside WSL (on Windows) or have your CUDA drivers installed correctly so TensorFlow can hijack the GPU. You can verify GPU detection by running: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
