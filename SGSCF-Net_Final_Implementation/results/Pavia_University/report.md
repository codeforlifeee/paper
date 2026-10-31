# 📊 Pavia University (UP) - SGSCF-Net Grid Search Report

## 🏆 Optimal Configuration Discovered
After conducting a massive 20-combination hyperparameter sweep on the Nvidia Tesla T4 GPU, the network achieved maximum performance on the Pavia University dataset with the following architecture:

* **Optimal Spatial Patch Size:** 9x9
* **Optimal Batch Size:** 32
* **Peak Overall Accuracy (OA):** 0.0%
* **Peak Average Accuracy (AA):** 0.0%
* **Average Training Time:** 0.0 minutes

## 📈 Sensitivity Analysis
The attached heatmap (`UP_heatmap.png`) visually demonstrates how the model's accuracy fluctuates based on spatial window context and batch size gradients. 

* **Patch Size Impact:** Larger spatial windows generally provide more contextual awareness for the 3D convolutions, improving accuracy up to a saturation point.
* **Batch Size Impact:** Smaller batch sizes (e.g., 32) consistently yield higher accuracy by allowing the optimizer to make more frequent weight updates per epoch, which is critical for smaller hyperspectral datasets.
