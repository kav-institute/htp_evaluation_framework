#### Reliability (avg RLS, min RLS) (Probabilistic): 
RLS measures the observed relative frequency of occurrences by calculating the estimated confidence levels for every corresponding pair of predicted distribution and ground truth point. If a prediction states that a pedestrian has a 90 % probability of being within a specified area, the prediction should occur 90 % of the time it is made. Otherwise, downstream tasks like path-planing cannot trust these predictions because the model is under- or overconfident. The metric provide a statistical guarantee of probabilistic and positional correctness. The average RLS score describes the mean reliability of all discrete forecast horizon timesteps, the minimum RLS score describes the worst occuring situation. 100 % indicates ideal calibration.

#### Sharpness (SS) (Probabilistic): 
Sharpness measures the volumetric measure of distribution confidence levels over time. It describes the average distribution spread size expansion over time.


#### Average Mahalanobis Distance (AMD) (Probabilistic): 
AMD tries to quantify how close all generated samples are to the ground truth using the Mahalanobis distance for measurement.

The AMD metric provides a single value that summarizes the average similarity or dissimilarity between two sets of points. A lower AMD value indicates that the sets are more similar, while a higher value indicates they are more dissimilar. 


#### Average Maximum Eigenvalue (AMV) (Probabilistic): 
AMV describes the overall spread of the predictions.

The AMV metric calculates the average of the maximum eigenvalues of the covariance matrices for each cluster. This provides a measure of the average dispersion or spread of the clusters in high-dimensional spaces.


#### Probability Cumulative Minimum Distance (PCMD) (Probabilistic): 
Metric to comprehensively and stably evaluate the learned model and latent distribution, which cumulatively selects the minimum distance between sampled trajectories and ground-truth trajectories from high probability to low probability by ranking. PCMD considers the predictions with corresponding probabilities and evaluates the prediction model under the whole latent distribution. Top-K selection is possible, Most-Likely evaluation uses ADE/FDE of predicition with highst probability. 

The authors guide to sample a total of K=80 samples from the models latent space, this should represent the whole distribution, which is quiet a bit optimistic. Therefore using PCMD with K=80 can not fully evaluate the complete distribution itself.


#### KDE-Negative-Log-Likelihood (KDE-NLL) (Probabilistic): 
Mean NLL of the ground truth trajectory under a distribution created by fitting a Kernel Density Estimation (KDE) on trajectory samples. The outcome is depending on fitting parameters and the number K of samples extracted from the model under test.

The metric itself only measures how good the predictied spread and mean fits with the GT data, but it does not evaluate the predicted spread itself: If the model predicts a trajectory with high uncertainty (wide spread of possible positions), but the actual position is within that range, the NLL will be lower. If the model predicts a trajectory with low uncertainty (a narrow spread of possible positions) but the actual position is far away from the predicted mean, the NLL will be higher.


#### Best of K min ADE/FDE @ Confidence Levels (Probabilistic): 
Having distributional data enables to derive certain confidence levels. Calculating the Best-of-K minimum ADE/FDE for different confidence levels delivers information about the overall correctness of the latent space density regarding the ground truth.


#### Best of K min ADE/FDE (Detemernistic): 
Classical Average- and Final-Displacement Error using the L2-distance between K predicted positions and ground truth, selecting the sample with the lowest average/final distance to the ground truth.


#### Joint ADE/FDE (JADE,JFDE) (Detemernistic): 
Joint-ADE/FDE expand the classical metrics regarding surrounding agents and possible collision scenarios, reducing unnatural predictions, i.e., colliding trajectories or diverging trajectories of groups. Achieved by swapping the order of taking the minimum over K samples and taking the average over N agents. This difference means we cannot mix-and-match agents between different samples; rather we must take the average error over
all agents within a sample before we select the best one to use in performance evaluation.


#### Average- and Final-Self-Distance (ASD/FSD) (Detemernistic):
One Ground Truth does not map multi-modality of possible future trajectories: ADE/FDE metrics do not penalize repeated samples. To address this, the ASD and FSD metrics are introduced to evaluate the similarity between generated samples in the set of (K) forecasted trajectories. Larger ASD and FSD scores mean the forecasted trajectories are more non-repetitive and diverse. 

ASD: Average L2 distance over all time steps between a forecasted sample and its closest neighbor sample
FSD: Final L2 distance over the last/final time step between a forecasted sample and its closest neighbor sample