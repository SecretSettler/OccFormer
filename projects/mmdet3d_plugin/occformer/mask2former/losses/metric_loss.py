import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from mmdet.models.builder import LOSSES
from projects.mmdet3d_plugin.occformer.lib.pointops.functions import pointops

class BayesianTripletLoss(nn.Module):

    def __init__(self, margin=0, varPrior=1/96.0, kl_scale_factor=1e-6, distribution='gauss'):
        super(BayesianTripletLoss, self).__init__()

        self.margin = torch.tensor(margin)
        self.varPrior = torch.tensor(varPrior)
        self.kl_scale_factor = kl_scale_factor
        self.distribution = distribution

    def forward(self, muA, muP, muN, varA, varP, varN):    # x:(D, 1+1+neg_count)

        muA, muP, muN, varA, varP, varN = muA.T, muP.T, muN.T, varA.T, varP.T, varN.T

        # calculate nll
        nll, probs, mu, sigma = negative_loglikelihood(muA, muP, muN, varA, varP, varN, margin=self.margin)

        # KL(anchor|| prior) + KL(positive|| prior) + KL(negative|| prior)
        if self.distribution == 'gauss':
            muPrior = torch.zeros_like(muA, requires_grad=False)
            varPrior = torch.ones_like(varA, requires_grad=False) * self.varPrior

            kl = (kl_div_gauss(muA, varA, muPrior, varPrior) + \
                kl_div_gauss(muP, varP, muPrior, varPrior) + \
                kl_div_gauss(muN, varN, muPrior, varPrior))

        elif self.distribution == 'vMF':
            kl = (kl_div_vMF(muA, varA) + \
            kl_div_vMF(muP, varP) + \
            kl_div_vMF(muN, varN))

        return nll + self.kl_scale_factor * kl, probs, mu, sigma


def negative_loglikelihood(muA, muP, muN, varA, varP, varN, margin=0.0):

    muA2 = muA**2                      # ([D-1, N])   # N = S - 2
    muP2 = muP**2                      # ([D-1, N])
    muN2 = muN**2                      # ([D-1, N])
    varP2 = varP**2                    # (1, N)
    varN2 = varN**2                    # (1, N)

    mu = torch.sum(muP2 + varP - muN2 - varN - 2 * muA * (muP - muN), dim=0)                                  # sum of feature dimension ([1, 5])
    T1 = varP2 + 2 * muP2 * varP + 2 * (varA + muA2) * (varP + muP2) - 2 * muA2 * muP2 - 4 * muA * muP * varP # ([2047, 5])
    T2 = varN2 + 2 * muN2 * varN + 2 * (varA + muA2) * (varN + muN2) - 2 * muA2 * muN2 - 4 * muA * muN * varN # ([2047, 5])
    T3 = 4 * muP * muN * varA                                                                                 # ([2047, 5])
    sigma2 = torch.sum(2 * T1 + 2 * T2 - 2 * T3, dim=0)                                                       # sum of feature dimension ([1, 5])
    # sigma = sigma2**0.5                                                                                       # ([1, 5])
    sigma = (sigma2+1e-7)**0.5                                                                                       # ([1, 5])

    # PyTorch uses a parametric CDF function that enables the gradient to flow back
    # try:
    probs = Normal(loc=mu, scale=sigma + 1e-8).cdf(margin)                     # ([1, 5])
    # except:
    #     print(f'mu: min={mu.min()} max={mu.max()}')
    #     print(f'sigma: min={sigma.min()} sigma={mu.max()}')
    nll = -torch.log(probs + 1e-8)                                             # ([1, 5])

    return nll.mean(), probs.mean(), mu.mean(), sigma.mean()


def kl_div_gauss(mu_q, var_q, mu_p, var_p):                # (D, N), (1, N)

    # N, D = mu_q.shape

    # kl diverence for isotropic gaussian
    # kl = 0.5 * ((var_q / var_p) * D + \
    #     1.0 / (var_p) * torch.sum(mu_p**2 + mu_q**2 - 2 * mu_p * mu_q, axis=1) - D + \
    #         D * (torch.log(var_p) - torch.log(var_q)))
    D, N = mu_q.shape

    kl = 0.5 * ((var_q / var_p) * D + 1.0 / (var_p) * torch.sum(mu_p**2 + mu_q**2 - 2 * mu_p * mu_q, axis=0) - D + D * (torch.log(var_p) - torch.log(var_q)))

    return kl.mean()


def kl_div_vMF(mu_q, var_q):
    N, D = mu_q.shape

    # we are estimating the variance and not kappa in the network.
    # They are propertional
    kappa_q = 1.0 / var_q
    kl = kappa_q - D * torch.log(2.0)

    return kl.mean()

@LOSSES.register_module()
class MetricLoss(nn.Module):

    def __init__(self, nsample=5, varPrior=1 / 96.0, kl_scale_factor=1e-6, margin=0):
        super().__init__()
        self.bayesian_triplet_loss = BayesianTripletLoss(varPrior=varPrior, kl_scale_factor=kl_scale_factor, margin=margin)
        self.nsample = torch.tensor(nsample)

    def forward(self, mask_pred, sigma, mask_target, pt_coords):
        '''
        mask_pred:        [N, 150528]
        sigma:            [N, 1]
        mask_target:      [N, 150528]
        pt_coords:        [N, 150528, 3]
        '''
        
        num_gts = mask_pred.shape[0]
        total_elements = mask_pred.shape[1]
        mask_pred = mask_pred.unsqueeze(-1)
        mask_target = mask_target.unsqueeze(-1)
        sigma = sigma.expand(num_gts, total_elements).unsqueeze(-1)
        neighbor_idx = pointops.knnquery(self.nsample, pt_coords, pt_coords) # (m, nsample)
        nsample = self.nsample - 1
        neighbor_idx = neighbor_idx[..., 1:].contiguous()
        neighbor_idx = neighbor_idx.squeeze(0)
        m = neighbor_idx.shape[0]
        neighbor_label = mask_target[:, neighbor_idx.view(-1).long(), :].view(num_gts, m, nsample, 1)       # (m, nsample, 1)
        neighbor_feature = mask_pred[:, neighbor_idx.view(-1).long(), :].view(num_gts, m, nsample, 1) # ([m, nsample, c])
        neighbor_sigma = sigma[:, neighbor_idx.view(-1).long(), :].view(num_gts, m, nsample, sigma.shape[2])       # ([m, nsample, c])

        posmask = mask_target == neighbor_label.view(num_gts, m, nsample) # (num_gt, m, nsample) - bool
        point_mask = torch.sum(posmask.int(), -1)  # (num_gt, m)
        point_mask = torch.logical_and(0 < point_mask, point_mask < nsample)

        posmask = posmask[point_mask] # ([num_gts, m', nsample]), boolen
        feature = mask_pred[point_mask]                 # ([num_gts, m', c])
        neighbor_feature = neighbor_feature[point_mask]    # ([num_gts, m', nsample, c])
        neighbor_sigma = neighbor_sigma[point_mask]        # ([num_gts, m', nsample, c])
        sigma = sigma[point_mask]# ([num_gts, m', 1])
       
        # TODO: choose more than 1 sample from neighbors
        pos_inds = torch.stack((torch.arange(posmask.shape[0]).flatten(), torch.randint(low=0, high=posmask.shape[1], size=(posmask.shape[0], 1)).flatten())).T
        pos_feats = neighbor_feature[pos_inds[:, 0], pos_inds[:, 1], :]
        pos_sigma = neighbor_sigma[pos_inds[:, 0], pos_inds[:, 1], :]

        neg_inds = torch.stack((torch.arange(posmask.shape[0]).flatten(), torch.randint(low=0, high=posmask.shape[1], size=(posmask.shape[0], 1)).flatten())).T
        neg_feats = neighbor_feature[neg_inds[:, 0], neg_inds[:, 1], :]
        neg_sigma = neighbor_sigma[neg_inds[:, 0], neg_inds[:, 1], :]

        pos_real = posmask[pos_inds[:, 0], pos_inds[:, 1]]
        neg_real = ~posmask[neg_inds[:, 0], neg_inds[:, 1]]
        tri_inds = torch.logical_and(pos_real, neg_real)    # all:327843 => boundary:58572 => triplet:9716

        loss, probs, t_mu, t_sigma2 = self.bayesian_triplet_loss(feature[tri_inds], pos_feats[tri_inds], neg_feats[tri_inds], sigma[tri_inds], pos_sigma[tri_inds], neg_sigma[tri_inds])
        meta = {'probs': probs, 't_mu': t_mu, 't_sigma2': t_sigma2}
        # return loss, meta
        return loss