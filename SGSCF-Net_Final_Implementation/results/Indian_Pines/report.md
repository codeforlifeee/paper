# 📊 Indian Pines (IN) - SGSCF-Net Grid Search Report

## 🏆 Optimal Configuration Discovered
After conducting a massive 20-combination hyperparameter sweep on the Nvidia Tesla T4 GPU, the network achieved maximum performance on the Indian Pines dataset with the following architecture:

* **Optimal Spatial Patch Size:** 11x11
* **Optimal Batch Size:** 32
* **Peak Overall Accuracy (OA):** 98.0519%
* **Peak Average Accuracy (AA):** 94.8249%
* **Average Training Time:** 8.8 minutes

## 📈 Sensitivity Analysis
The attached heatmap (`IN_heatmap.png`) visually demonstrates how the model's accuracy fluctuates based on spatial window context and batch size gradients. 

* **Patch Size Impact:** Larger spatial windows generally provide more contextual awareness for the 3D convolutions, improving accuracy up to a saturation point.
* **Batch Size Impact:** Smaller batch sizes (e.g., 32) consistently yield higher accuracy by allowing the optimizer to make more frequent weight updates per epoch, which is critical for smaller hyperspectral datasets.
