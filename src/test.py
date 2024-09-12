import os
import sys
import numpy as np
import h5py
import json

from traj_eval import HTP_Eval
from loader import Numpy_Loader, Pth_Loader, H5_Loader

FW = 'social_implicit'
TAG = 'univ'
WORK_DIR = '/workspace/data'

cfg = {}
cfg['device'] = 'cpu'
cfg['delta_t'] = 0.4
cfg['n_features'] = 2
cfg['n_components'] = 3
cfg['n_forecast_horizons'] = 12
cfg['mesh_resolution'] = 0.1
cfg['confidence_levels'] = [0.995, 0.95, 0.68]
cfg['pcmd_levels'] = [1,5,20]
cfg['dest_dir'] = f'{WORK_DIR}/{FW}/{TAG}/results'

input_path = f'{WORK_DIR}/{FW}/{TAG}/{TAG}_input.npy'
gt_path = f'{WORK_DIR}/{FW}/{TAG}/{TAG}_gt.npy'
k_samples_path = f'{WORK_DIR}/{FW}/{TAG}/{TAG}_k_samples_.npy'
full_samples_path = f'{WORK_DIR}/{FW}/{TAG}/{TAG}_pred_samples_.npy'
gmms_path = f'{WORK_DIR}/{FW}/{TAG}/{TAG}_pred_gmms_.npz'

# Numpy loader
loader = Numpy_Loader(input_path=input_path, gt_path=gt_path, k_samples_path=k_samples_path, full_samples_path=full_samples_path, gmms_path=gmms_path)
input = loader.load_input()
gt = loader.load_gt()
full_samples = loader.load_samples()

torch_gmms = loader.fit_gmms_to_samples(samples=full_samples, method='em', n_components=cfg['n_components'], n_forecast_horizons=cfg['n_forecast_horizons'], n_iters=500, n_parallel=12)
loader.write_gmms(torch_gmms, path=gmms_path)

torch_gmms = loader.load_gmms()



# # Pth loader
# loader = Pth_Loader(input_path=input_path, gt_path=gt_path, k_samples_path=k_samples_path, full_samples_path=full_samples_path, gmms_path=gmms_path)
# input = loader.load_input()
# gt = loader.load_gt()
# full_samples = loader.load_samples()

# torch_gmms = loader.fit_gmms_to_samples(samples=full_samples, method='em', n_components=cfg['n_components'], n_forecast_horizons=cfg['n_forecast_horizons'], n_iters=500, n_parallel=12)
# loader.write_gmms(torch_gmms, path=gmms_path)

# torch_gmms = loader.load_gmms()


# # H5 loader
# loader = H5_Loader(input_path=input_path, gt_path=gt_path, k_samples_path=k_samples_path, full_samples_path=full_samples_path, gmms_path=gmms_path)
# input = loader.load_input()
# gt = loader.load_gt()
# full_samples = loader.load_samples()

# torch_gmms = loader.fit_gmms_to_samples(samples=full_samples, method='em', n_components=cfg['n_components'], n_forecast_horizons=cfg['n_forecast_horizons'], n_iters=500, n_parallel=12)
# loader.write_gmms(torch_gmms, path=gmms_path)

# torch_gmms = loader.load_gmms()





# setup
results = []
eval = HTP_Eval(cfg=cfg, input=input, gt=gt, full_samples=full_samples, torch_gmms=torch_gmms)

# # calc metrices
results.append(eval.get_ade_fde(k=20))
results.append(eval.get_ade_fde_for_cls(k=20))
results.append(eval.get_rls(with_plot=True))
results.append(eval.get_ss(n_parallel=2))
results.append(eval.get_amd_amv(n_parallel=1))
results.append(eval.get_kde_nll(n_parallel=1))
results.append(eval.get_pcmd(k=80, with_plot=True))

# save scores
result_dict = eval.combine_dicts(dict_list=results)
with open(os.path.join(cfg['dest_dir'], 'results.json'), 'w') as f: json.dump(result_dict, f)

# # vis prediction confidence levels
eval.vis_predictions(n_parallel=8, step=32)

# vis gmm fit results
eval.vis_gmm_fits(n_parallel=8, step=32)

sys.exit()