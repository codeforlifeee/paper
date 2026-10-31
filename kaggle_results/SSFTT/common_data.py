import os, scipy.io as sio, glob as _g

def get_hsi_data_dir():
    for c in ['/kaggle/input/dataset-mp',
              '/kaggle/input/datasets/tejassurya2004/dataset-mp',
              '/kaggle/input/minor-project-data']:
        if os.path.exists(os.path.join(c, 'Indian_pines_corrected.mat')):
            return os.path.abspath(c)
    hits = _g.glob('/kaggle/input/**/Indian_pines_corrected.mat', recursive=True)
    return os.path.dirname(hits[0]) if hits else '/kaggle/input/dataset-mp'

def load_indian_pines():
    d = get_hsi_data_dir()
    return (
        sio.loadmat(os.path.join(d, 'Indian_pines_corrected.mat'))['indian_pines_corrected'],
        sio.loadmat(os.path.join(d, 'Indian_pines_gt.mat'))['indian_pines_gt']
    )
