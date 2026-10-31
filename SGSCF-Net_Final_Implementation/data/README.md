# Datasets

The three hyperspectral benchmarks are **not bundled** with this repository
(they are large and distributed by their original hosts). Download the `.mat`
files and place them directly in this `data/` folder.

| Dataset | Required files | Source |
| --- | --- | --- |
| Indian Pines | `Indian_pines_corrected.mat`, `Indian_pines_gt.mat` | [GIC / Universidad del País Vasco](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| Pavia University | `PaviaU.mat`, `PaviaU_gt.mat` | same source as above |
| Salinas | `Salinas_corrected.mat`, `Salinas_gt.mat` | same source as above |

After download, the folder should look like:

```
data/
├── Indian_pines_corrected.mat
├── Indian_pines_gt.mat
├── PaviaU.mat
├── PaviaU_gt.mat
├── Salinas_corrected.mat
└── Salinas_gt.mat
```

The training script auto-detects this folder. To use a different location,
pass `--data_dir /path/to/data` on the command line.
