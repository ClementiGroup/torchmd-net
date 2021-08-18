import torch
import torch.nn as nn
import math
import opt_einsum as oe
from ..spherical_expansion import SphericalExpansion
from typing import List, Union

def powerspectrum(se_, nsp, nmax, lmax):
    J = se_.shape[0]
    se = se_.view((J, nsp, nmax, lmax**2))
    ps = torch.zeros(J, nsp, nmax, nsp, nmax, lmax, dtype=se.dtype, device=se.device)
    idx = 0
    for l in range(lmax):
        lbs = 2*l+1
        ps[..., l] = torch.sum(torch.einsum('ianl,ibml->ianbml', se[..., idx:idx+lbs], se[..., idx:idx+lbs]),
                                   dim=5) / math.sqrt(lbs)
        idx += lbs
    ps = ps.view(J, nsp* nmax, nsp* nmax, lmax)
    PS = torch.zeros(J, int((nsp* nmax+1)**2/2), lmax, dtype=se.dtype, device=se.device)

    fac = math.sqrt(2.) * torch.ones((nsp*nmax,nsp*nmax))
    fac[range(nsp*nmax), range(nsp*nmax)] = 1.
    ids = [(i,j) for i in range(nsp*nmax)
                for j in range(nsp*nmax) if j >= i]
    for ii, (i, j) in enumerate(ids):
            PS[:, ii, :] = fac[i,j]*ps[:, i, j, :]
    return PS.view(J, -1)

def powerspectrum_(se_, nsp, nmax, lmax):
    J = se_[0].shape[0]
    dtype = se_[0].dtype
    device = se_[0].device

    se = []
    for l in range(lmax):
        se.append(se_[l].view(J, nsp* nmax, 2*l+1))

    ps = torch.zeros(lmax, J, nsp* nmax, nsp* nmax, dtype=dtype, device=device)
    for l in range(lmax):
        ps[l] = oe.contract('iam,ibm->iab', se[l], se[l]) / math.sqrt(2*l+1)

    # ps = ps.view(lmax, J, nsp* nmax, nsp* nmax)
    PS = torch.zeros(J, lmax, int((nsp* nmax+1)**2/2), dtype=dtype, device=device)

    fac = math.sqrt(2.) * torch.ones((nsp*nmax,nsp*nmax))
    fac[range(nsp*nmax), range(nsp*nmax)] = 1.
    ids = [(i,j) for i in range(nsp*nmax)
                for j in range(nsp*nmax) if j >= i]
    for ii, (i, j) in enumerate(ids):
            PS[:, :, ii] = fac[i,j]*ps[:,:, i, j].transpose(0,1)
    return PS.view(J, -1)

class PowerSpectrum(nn.Module):
    def __init__(self, max_radial: int, max_angular: int,
                    interaction_cutoff: float,
                                gaussian_sigma_constant: float, species: Union[List[int], torch.Tensor], normalize: bool =True, smooth_width: float=0.5):
        super(PowerSpectrum, self).__init__()
        self.nmax = max_radial
        self.lmax = max_angular
        self.rc = interaction_cutoff
        self.sigma = gaussian_sigma_constant
        self.normalize = normalize
        if isinstance(species, list):
            species = torch.tensor(species, dtype=torch.long)
        self.species, _ = torch.sort(species)

        self.n_species = len(species)
        self.species2idx = -1*torch.ones(torch.max(species)+1,dtype=torch.long)
        for isp, sp in enumerate(self.species):
            self.species2idx[sp] = isp

        self.se = SphericalExpansion(max_radial, max_angular, interaction_cutoff, gaussian_sigma_constant, species, smooth_width=smooth_width)

        self.D = int((self.n_species*self.nmax+1)**2/2) * (self.lmax+1)

    def size(self):
        return int((self.n_species*self.nmax+1)**2/2) * (self.lmax+1)

    def forward(self, data):
        # ci_anlm = self.se(data)
        # pi_anbml = powerspectrum(ci_anlm, self.n_species,
        #                                 self.nmax, self.lmax+1)
        cl_ianm = self.se(data)
        pi_anbml = powerspectrum_(cl_ianm, self.n_species,
                                        self.nmax, self.lmax+1)
        if self.normalize:
            return torch.nn.functional.normalize(
                pi_anbml.view(-1, self.D), dim=1)
        else:
            return pi_anbml.view(-1, self.D)
