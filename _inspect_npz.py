import numpy as np

data = np.load('D:/gaze/vit-human-attention-comparison/dataset/Nakano_etal_2010/preprocessed_data/vit_gaze_pos_author.npz', allow_pickle=True)
print('Keys:', list(data.keys()))
for k in data.keys():
    v = data[k]
    if hasattr(v, 'shape'):
        print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
    else:
        print(f'  {k}: type={type(v)}, value={v}')
