import os
import sys
import torch
import numpy as np
import seaborn as sns

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from functools import reduce
from termcolor import colored
from joblib import Parallel, delayed
from tqdm import tqdm
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily

from ade_fde import calc_ade_fde, calc_ade_fde_for_cls
from rls_ss import build_confidence_set, build_confidence_levels, build_mesh_grid, calc_reliability, calc_sharpness_sets, calc_sharpness
from amd_amv import calc_amd_amv
from kde import calc_kde
from pcmd import calc_pcmd
from asd_fsd import calc_asd_fsd #TODO: not yet implemented
from jade_jfde import calc_jade_jfde #TODO: not yet implemented

sys.path.append("/workspace/repos/src")

# Additional parameters for visualization
CONF_LEVEL_APLHA = 0.7
SAMPLE_POINTS_ALPHA = 0.4


class HTP_Eval():
    
    def __init__(self, cfg, input, gt, full_samples, torch_gmms):
        
        self.input = input
        self.gt = gt
        self.full_samples = full_samples
        self.gmms = torch_gmms
        
        self.n_tracks = self.full_samples.shape[0]
        self.n_sample_points = self.full_samples.shape[1]
        self.n_features = cfg['n_features']
        self.n_components = cfg['n_components']
        self.device = cfg['device']
        self.n_forecast_horizons = cfg['n_forecast_horizons']
        self.work_dir = cfg['dest_dir'] 
        
        self.confidence_levels = cfg['confidence_levels']
        self.pcmd_levels = cfg['pcmd_levels']
        self.mesh_resolution = cfg['mesh_resolution']
        self.delta_t = cfg['delta_t']
        
        self.confidence_level_alpha = CONF_LEVEL_APLHA
        self.sample_points_alpha = SAMPLE_POINTS_ALPHA
        
        self.bins = [k for k in np.arange(0.0, 1.01, 0.01)]
        self.colors_palette = np.array(sns.color_palette(palette='Spectral_r', n_colors=self.n_forecast_horizons))
        if not os.path.exists(self.work_dir): os.makedirs(self.work_dir)
        
        return
    
    
    def get_ade_fde(self, k):
        """manage min ade/fde calulation
        Args:
            k (int): number k of samples to get from distributions
        Returns:
            dict: ade/fde scores
        """
        
        res_dict = {}
        
        # pytorch batched processing
        ade, fde = calc_ade_fde(gt=self.gt, gmm=self.gmms, k=k)
        
        res_dict[f"min_ADE (K={k}):"] = ade
        res_dict[f"min_FDE (K={k}):"] = fde
        return res_dict
    
    
    def get_ade_fde_for_cls(self, k):
        """manage min ade/fde calulation at defined confidence levels
        Args:
            k (int): number k of samples to get from distributions
        Returns:
            dict: ade/fde scores
        """
        
        # pytorch batched processing
        res = calc_ade_fde_for_cls(gt=self.gt, 
                            gmm=self.gmms, 
                            n_sample_points=self.n_sample_points,
                            k=k,
                            n_forecast_horizons=self.n_forecast_horizons,
                            n_features=self.n_features,
                            confidence_levels=self.confidence_levels)
        
        return res
    
    
    def get_rls(self, with_plot):
        """manage reliability score calculation
        Args:
            with_plot (bool): with or without reliability calibration plot
        Returns:
            dict: min and avg rls scores
        """
        
        # pytorch batched processing
        res = calc_reliability(gt=self.gt, 
                                gmm=self.gmms, 
                                n_sample_points=self.n_sample_points, 
                                percentiles=self.bins, 
                                n_forecasts_horizons=self.n_forecast_horizons,
                                dt=self.delta_t, 
                                dir=self.work_dir, 
                                with_plot=with_plot
                                )
        
        return res
    
    
    def get_ss(self, n_parallel=1):
        """manage reliability score calculation
        Args:
            n_parallel (int): number of parallel multiprocessing processes
        Returns:
            dict: ss scores at defined confidence levels
        """
        
        print(colored(f" - Calculating sharpness scores", 'green'))
        
        # do for every track, multiprocessing is available
        sharpness_sets = list(tqdm(Parallel(return_as="generator", n_jobs=n_parallel)(delayed(calc_sharpness_sets)(gmm=self.get_gmm_by_index(idx=idx),  
                                                                                                            samples=self.full_samples, 
                                                                                                            n_sample_points=self.n_sample_points,
                                                                                                            mesh_resolution=self.mesh_resolution,
                                                                                                            confidence_levels=self.confidence_levels)
                                                                                                            for idx in range(self.n_tracks)), total=self.n_tracks))
        
        res = calc_sharpness(sharpness_sets=torch.stack(sharpness_sets).numpy(), 
                            confidence_levels=self.confidence_levels, 
                            percentiles=self.bins,
                            n_forecasts_horizons=self.n_forecast_horizons, dt=self.delta_t)
        
        return res
    
    
    def get_amd_amv(self, n_parallel=1):
        """manage amd/amv score calculation
        Args:
            n_parallel (int): number of parallel multiprocessing processes
        Returns:
            dict: amd/amv scores
        """
        
        print(colored(f" - Calculating amd/amv scores", 'green'))
        
        # do for every track, multiprocessing is available for larger batches
        res = list(tqdm(Parallel(return_as="generator", n_jobs=n_parallel)(delayed(calc_amd_amv)(gt=self.gt[idx], 
                                                                                        gmm=self.get_gmm_by_index(idx=idx), 
                                                                                        n_forecasts_horizons=self.n_forecast_horizons) 
                                                                                        for idx in range(self.n_tracks)), total=self.n_tracks))
        
        # get means
        res_dict = {}
        amd = np.array(res)[...,0].mean()
        amv = np.array(res)[...,1].mean()
        score = (amd+amv)/2
        
        # save
        res_dict[f"AMD:"] = amd
        res_dict[f"AMV:"] = amv
        res_dict[f"(AMD+AMV)/2:"] = score
        return res_dict
    
    
    def get_kde_nll(self, n_parallel=1):
        """manage kde-nll calculation
        Args:
            n_parallel (int): number of parallel multiprocessing processes
        Returns:
            dict: kde score
        """
        
        print(colored(f" - Calculating kde-nll scores", 'green'))
        
        # do for every track, multiprocessing is available for larger batches
        res = list(tqdm(Parallel(return_as="generator", n_jobs=n_parallel)(delayed(calc_kde)(gt=self.gt[idx], 
                                                                                    samples=self.full_samples[idx], 
                                                                                    n_forecasts_horizons=self.n_forecast_horizons) 
                                                                                    for idx in range(self.n_tracks)), total=self.n_tracks))
        
        res_dict = {}
        kde = np.array(res).mean()
        res_dict[f"KDE-NLL:"] = kde
        
        return res_dict
    
    
    def get_pcmd(self, k, with_plot):
        """calculate pcmd scores, batched
        Args:
            k (int): number k of samples to get from distributions
            with_plot (bool): with or without plot
        Returns:
            dict: pcmd scores
        """
        
        # pytorch batched processing
        res = calc_pcmd(gt=self.gt, gmm=self.gmms, k=k, pcmd_levels=self.pcmd_levels, dir=self.work_dir, with_plot=with_plot) 
        
        return res
    
    
    def vis_predictions(self, n_parallel=1, step=1):
        
        print(colored(f" - Visualize predictions", 'green'))
        
        p = os.path.join(self.work_dir, 'predictions')
        if not os.path.exists(p):  os.makedirs(p)
        
        _ = list(tqdm(Parallel(return_as="generator", n_jobs=n_parallel)(delayed(self.vis_single_prediction)(idx=idx, 
                                                                                                                p=p) 
                                                                                                                for idx in range(0, self.n_tracks, step)), total=int(self.n_tracks/step)))
            
        return
    
    
    def vis_single_prediction(self, idx, p):
        
        input = self.input[idx].unsqueeze(0)
        gt = self.gt[idx].unsqueeze(0)
        points = self.full_samples[idx]
        torch_gmm = self.get_gmm_by_index(idx=idx)
        grid, steps, _, r = build_mesh_grid(samples=points, mesh_resolution=self.mesh_resolution)
        confidence_map = build_confidence_set(gt=grid, gmm=torch_gmm, n_sample_points=self.n_sample_points)
        modes = grid[torch.argmin(confidence_map, dim=0, keepdim=True)][0,:,0,:]
        confidence_areas = build_confidence_levels(confidence_map=confidence_map, modes=modes, confidence_levels=self.confidence_levels, n_forecast_horizons=self.n_forecast_horizons, steps=steps, mesh_resolution=self.mesh_resolution, r=r)
        self.plot_predictions(input=input.numpy(), gt=gt.numpy(), forecasts=confidence_areas, modes=modes.numpy(), dst_dir=p, sample_id=idx)
        
        return
    
    
    def plot_predictions(self, input, gt, forecasts, modes, dst_dir, sample_id):
        """Plot complete forecast
        """
        
        # dst dir and epoch handling
        if not os.path.exists(dst_dir): os.makedirs(dst_dir)
            
        input_data = input.squeeze(0).tolist()
        gt_data = gt.squeeze(0).tolist()
        mode_data = modes.tolist()
        
        # create figure
        fig, axs = plt.subplots(ncols=len(self.confidence_levels), nrows=1, figsize=(19.2, 8.0), sharex=True, sharey=True)
        axs = axs.flatten()  # Flatten the array of axes objects
        
        # plot confidence areas
        for idx, cl in enumerate(self.confidence_levels):
            
            for k, contours in enumerate(reversed(forecasts[idx])):
                
                color = self.colors_palette[self.n_forecast_horizons-k-1]
                
                for c in contours:
                    
                    polygon = plt.Polygon(c, facecolor=color, edgecolor=color, alpha=0.40)
                    axs[idx].add_patch(polygon)
                    
            # plot most likely positions
            for l, m in enumerate(reversed(mode_data)):
                
                color = self.colors_palette[self.n_forecast_horizons-l-1]
                label = 'Most likely @ ' + str(round((self.n_forecast_horizons - l) * self.delta_t, 1)) + ' s'
                axs[idx].plot(m[0], m[1], marker='.', markersize=12, color=color, label=label, markeredgecolor=color, markeredgewidth=1)
                    
            # plot input and gt data
            axs[idx].plot(np.array(input_data)[...,0], np.array(input_data)[...,1], color='r', linewidth=2, label="Input", marker='.', markersize=10)
            axs[idx].plot(np.array(gt_data)[...,0], np.array(gt_data)[...,1], color='k', linewidth=2, label="Ground Truth", marker='*', markersize=10)
            
            axs[idx].set_aspect('equal', 'box')
            axs[idx].grid()
            axs[idx].set_title(f'Prediction @ {cl*100} % confidence level')
            axs[idx].set_xlabel("x / m", fontsize = 12)
            axs[idx].set_ylabel("y / m", fontsize = 12)
        
        plt.suptitle(f'Track ID: {sample_id}')
        fig.tight_layout()
        plt.savefig(os.path.join(dst_dir, f'sample_{str(sample_id).zfill(8)}.png'))
        plt.close()
        
        return
    
    
    def vis_gmm_fits(self, n_parallel=1, step=1):
        
        print(colored(f" - Visualize mixture fits", 'green'))
        
        p = os.path.join(self.work_dir, 'gmm_fits')
        if not os.path.exists(p):  os.makedirs(p)
        
        _ = list(tqdm(Parallel(return_as="generator", n_jobs=n_parallel)(delayed(self.vis_single_gmm_fit)(idx=idx, p=p) for idx in range(0, self.n_tracks, step)), total=int(self.n_tracks/step)))
        
        return
    
    
    def vis_single_gmm_fit(self, idx, p):
        
        points = self.full_samples[idx]
        torch_gmm = self.get_gmm_by_index(idx=idx)
        grid, steps, _, r = build_mesh_grid(samples=points, mesh_resolution=self.mesh_resolution)
        confidence_map = build_confidence_set(gt=grid, gmm=torch_gmm, n_sample_points=self.n_sample_points)
        modes = grid[torch.argmin(confidence_map, dim=0, keepdim=True)][0,:,0,:]
        confidence_areas = build_confidence_levels(confidence_map=confidence_map, modes=modes, confidence_levels=self.confidence_levels, n_forecast_horizons=self.n_forecast_horizons, steps=steps, mesh_resolution=self.mesh_resolution, r=r)
        self.plot_gmm_fits(sample_points=points.numpy(), forecasts=confidence_areas, dst_dir=p, sample_id=idx)
        
        return
    
    
    def plot_gmm_fits(self, sample_points, forecasts, dst_dir, sample_id):
        
        # Calculate the number of rows and columns needed
        num_plots = self.n_forecast_horizons
        num_cols = 4
        num_rows = int(np.ceil(num_plots / num_cols))
        
        # Shift points towards coordinate origin for better visualization
        shifted_sample_points = sample_points - np.mean(sample_points, axis=(0), keepdims=True)
        x_min, x_max, y_min, y_max = self.get_limits(arr=shifted_sample_points)
        
        # Create a figure with enough space for all subplots
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(19.2, 4 * num_rows), sharex=True, sharey=True)
        axs = axs.flatten()
        
        # Iterate over data and plot each dataset in its own subplot
        for fh, ax in enumerate(axs):
            
            # plot full confidence area
            color = self.colors_palette[fh]
            ax.scatter(shifted_sample_points[...,fh,0], shifted_sample_points[...,fh,1], color='k', marker='.', s=8, alpha=self.sample_points_alpha)
            
            # plot confidence areas
            for idx, _ in enumerate(self.confidence_levels):
                
                for c in forecasts[idx][fh]:
                    
                    shifted_c = c - np.mean(c, axis=0, keepdims=True)
                    alpha = (idx+1)*(self.confidence_level_alpha/len(self.confidence_levels))
                    polygon = plt.Polygon(shifted_c, facecolor=color, edgecolor=color, alpha=alpha)
                    ax.add_patch(polygon)
            
            ax.set_box_aspect(aspect=1)
            ax.grid()
            ax.set_title('@ ' + str(round((fh+1) * self.delta_t, 1)) + ' s')
            ax.set_xlabel("x / m", fontsize = 12)
            ax.set_ylabel("y / m", fontsize = 12)
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            
        # Hide any unused subplots
        for ax in axs[num_plots:]:
            ax.axis("off")
            
        plt.suptitle(f'GMM fits for sample points at different confidence levels: {self.confidence_levels}', fontsize=20)
        fig.tight_layout()
        plt.savefig(os.path.join(dst_dir, f'sample_{str(sample_id).zfill(8)}.png'))
        plt.close()
        
        return
    
    
    def get_gmm_by_index(self, idx):
        """extract single mixture object from batched representation
        Args:
            idx (int): index of element to extract from batched mixture object including all tracks
        Returns:
            torch.distribution.MixtureSameFamily: single mixture object
        """
        
        weights = self.gmms.mixture_distribution.probs[idx].unsqueeze(0)
        means = self.gmms.component_distribution.loc[idx].unsqueeze(0)
        covariances = self.gmms.component_distribution.covariance_matrix[idx].unsqueeze(0)
        
        normal = MultivariateNormal(means, covariance_matrix=covariances)
        mixture = Categorical(probs=weights)
        torch_gmm = MixtureSameFamily(mixture, normal)
        
        return torch_gmm
    
    
    def round_up(self, x):
        
        return np.ceil(np.abs(x)) * np.sign(x)
    
    
    def combine_dicts(self, dict_list):
        
        return reduce(lambda a, b: {**a, **b}, dict_list)
    
    
    def get_limits(self, arr):
        
        # Reshape the array to have a single column for each coordinate
        coords = arr.reshape(-1, 2)
        
        # Calculate minimum and maximum values for x and y coordinates
        x_min = np.floor(np.min(coords[:, 0])) if np.any(np.min(coords[:, 0]) < 0) else np.ceil(np.min(coords[:, 0]))
        x_max = np.floor(np.max(coords[:, 0])) if np.any(np.max(coords[:, 0]) < 0) else np.ceil(np.max(coords[:, 0]))
        y_min = np.floor(np.min(coords[:, 1])) if np.any(np.min(coords[:, 1]) < 0) else np.ceil(np.min(coords[:, 1]))
        y_max = np.floor(np.max(coords[:, 1])) if np.any(np.max(coords[:, 1]) < 0) else np.ceil(np.max(coords[:, 1]))
        
        return x_min, x_max, y_min, y_max