from torchmdnet.models import create_model
from torch_geometric.data import DataLoader
from datetime import datetime
from cgnet.feature import GeometryFeature
from torchmdnet2.nn import BaselineModel
from torchmdnet2.nn import CGnet
from torchmdnet2.simulation import Simulation
import json
import os
import torch.nn as nn
import torch
import pickle as pkl
import numpy as np
import sys
from cgnet.network import RepulsionLayer


KBOLTZMANN = 1.38064852e-23
AVOGADRO = 6.022140857e23
JPERKCAL = 4184

try:
    with open(sys.argv[1]) as json_file:
        sim_opts = json.load(json_file)
except:
    raise RuntimeError('Please supply simulation parameter file.')


peptide_ids = sys.argv[2:]

NETWORK_MODEL = sim_opts['network_model']
ARGS = sim_opts['args']
DEVICE_ID = sim_opts['device_id']
N_SIMS = sim_opts['n_sims']
N_TIMESTEPS = sim_opts['n_timesteps']
SAVE_INTERVAL = sim_opts['save_interval']
EXPORT_INTERVAL = sim_opts['export_interval']
LOG_INTERVAL = sim_opts['log_interval']
DT = sim_opts['dt']
FRICTION = sim_opts['friction']
TEMPERATURE = sim_opts['temperature']
MASS_SCALE = sim_opts['mass_scale']
SAVE_DIR = sim_opts['save_dir']
TAG = sim_opts['tag']
DEVICE = torch.device('cuda', DEVICE_ID)
BETA = JPERKCAL/KBOLTZMANN/AVOGADRO/TEMPERATURE
PRIOR_DICTIONARY_TAG = sim_opts['prior_dictionary_tag']

peptide_ids = [int(i) for i in peptide_ids]

for PEPTIDE_ID in peptide_ids:
    print("Simulating peptide {}...".format(PEPTIDE_ID))
    peptide_dictionary = pkl.load(open('/net/data02/nickc/peptide_cg_meta_dictionary_no_physical_GLY.pkl', 'rb'))
    ref_coords = np.load('/net/data02/nickc/octapeptides/brooke_map_cg_data/opep_{:04d}_cg_coords_no_physical_GLY.npy'.format(PEPTIDE_ID))
    embeddings = peptide_dictionary[PEPTIDE_ID]['embeddings']
    prior_info = pkl.load(open('/net/data02/nickc/octapeptides/brooke_map_cg_data/p{}_{}.pkl'.format(PEPTIDE_ID, PRIOR_DICTIONARY_TAG), 'rb'))
    prior_set = prior_info['priors']
    all_features = prior_info['all_features']
    distance_idx = prior_info['distance_idx']
    resmap = peptide_dictionary[PEPTIDE_ID]['resmap']

    geom_feat = GeometryFeature(feature_tuples=all_features,device=DEVICE)
    priors = []
    for prior in prior_set:
        priors += [prior.to(DEVICE)]

    with open(ARGS, "rb") as argfile:
        args = pkl.load(argfile)
    with open(NETWORK_MODEL, "rb") as modelfile:
        state_dict = pkl.load(modelfile)
    net_model = create_model(args)
    net_model.load_state_dict(state_dict)
    baseline = BaselineModel(geom_feat, priors, n_beads=len(embeddings))
    full_model = CGnet(net_model, baseline)
    full_model.to(DEVICE)

    posresidues = [(i, resmap[i]) for i in range(1, len(resmap)+1)]
    MASSES = []
    for posres in posresidues:
        i, resname = posres
        if resname != 'GLY':
            MASSES.append(14)
            MASSES.append(12)
            MASSES.append(12)
            MASSES.append(12)
            MASSES.append(16)
        else:
            MASSES.append(14)
            MASSES.append(12)
            MASSES.append(12)
            MASSES.append(16)

    MASSES = np.array(MASSES)

    starting_idx = np.random.choice(np.arange(len(ref_coords)), replace=False,size=(N_SIMS,))
    initial_coords = torch.tensor(ref_coords[starting_idx],
                                  requires_grad=True,
                                  dtype=torch.float).to(DEVICE)
    embeddings = torch.tensor(np.tile(embeddings, [N_SIMS, 1])).to(DEVICE)

    mysim = Simulation(full_model, initial_coords, embeddings,
                     length=N_TIMESTEPS, save_interval=SAVE_INTERVAL,
                     friction=FRICTION, masses=MASSES/MASS_SCALE,
                     beta=BETA, dt=DT, save_potential=True,
                     save_forces=True, device=DEVICE,
                     export_interval=EXPORT_INTERVAL,
                     log_interval=LOG_INTERVAL,
                     log_type='write',
                     filename='{}pep_{}_{}'.format(SAVE_DIR, PEPTIDE_ID, TAG))

    mysim.simulate()
    np.save('{}pep_{}_{}_starting_idx.npy'.format(SAVE_DIR, PEPTIDE_ID, TAG, starting_idx), starting_idx)
    del mysim
