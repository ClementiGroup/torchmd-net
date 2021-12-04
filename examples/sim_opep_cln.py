from torchmdnet.models.torchmd_gn import TorchMD_GN
from torchmdnet.simulation import Simulation
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import pickle as pkl
import sys
import json

KBOLTZMANN = 1.38064852e-23
AVOGADRO = 6.022140857e23
JPERKCAL = 4184

try:
    with open(sys.argv[1]) as json_file:
        sim_opts = json.load(json_file)
except:
    raise RuntimeError('Please supply simulation parameter file.')

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

np.random.seed(1875432)
torch.manual_seed(1875432)

# Meta information about CLN in the OPEPS mapping
meta_dictionary = pkl.load(open('/import/a12/users/nickc/cln_for_octapeptides/cln_meta_dictionary_no_physical_GLY.pkl', 'rb'))

# lets load the relevant prior/feature information from the prior dictionary.
prior_info = pkl.load(open('/import/a12/users/nickc/cln_for_octapeptides/cln_{}.pkl'.format(PRIOR_DICTIONARY_TAG), 'rb'))
prior_set = prior_info['priors']
all_features = prior_info['all_features']
distance_idx = prior_info['distance_idx']

# load reference coords and set embeddings
ref_coords = np.load('/import/a12/users/nickc/cln_for_octapeptides/opep_mapping_cln_coords_no_physical_GLY.npy')

from torchmdnet.feature import GeometryFeature
from torchmdnet.models import create_model
from torchmdnet.models.priors import BaselineModel
from torchmdnet.models.cgnet import CGnet

# Prepare the BaselineModel 
geom_feat = GeometryFeature(feature_tuples=all_features,device=DEVICE)
priors = []
for prior in prior_set:
    priors += [prior.to(DEVICE)]
n_beads = ref_coords.shape[1]
baseline = BaselineModel(geom_feat, priors, n_beads=n_beads)

# Load the network model
with open(ARGS, "rb") as argfile:
    args = pkl.load(argfile)
with open(NETWORK_MODEL, "rb") as modelfile:
    state_dict = pkl.load(modelfile)
net_model = create_model(args)
net_model.load_state_dict(state_dict)

# Switch the model max_num_neighbors to something greater than 40 to accomodate CLN
# This is a static (non-trainable parameter for torch_cluster.radius_graph, wrapped in a class the computes neighbors) - its ok to change it
from torch_cluster import radius_graph
from torchmdnet.models.utils import Distance

print(net_model)
net_model.representation_model.distance = Distance(cutoff_lower=args["cutoff_lower"], cutoff_upper=args["cutoff_upper"], max_num_neighbors=50)
print(net_model.representation_model.distance.max_num_neighbors)
net_model.to(DEVICE)

# Combine network and baseline
full_model = CGnet(net_model, baseline)

# Generate mass array
resmap = meta_dictionary["resmap"]
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

# Simulation settings
n_sims = 50
n_timesteps = 1000000
save_interval = 10
export_interval = 100000
log_interval = 100000
dt = 0.004 #fs
friction = 1 #inverse ps
temperature = 300 # K
beta =  1.677399001146518
masses = np.array([
        14, 12, 12, 12, 16, # TYR 
        14, 12, 12, 12, 16, # TYR
        14, 12, 12, 12, 16, # ASP
        14, 12, 12, 12, 16, # PRO
        14, 12, 12, 12, 16, # GLU
        14, 12, 12, 12, 16, # THR
        14, 12, 12, 16,     # GLY
        14, 12, 12, 12, 16, # THR
        14, 12, 12, 12, 16, # TRP
        14, 12, 12, 12, 16  # TYR
    ])

starting_coords = np.random.choice(np.arange(len(ref_coords)), replace=False, size=(n_sims,))
initial_coords = torch.tensor(ref_coords[starting_coords],
                              requires_grad=True,
                              dtype=torch.float).to(DEVICE)
embeddings = torch.tensor(np.tile(meta_dictionary['embeddings'], [n_sims, 1])).to(DEVICE)

mysim = Simulation(full_model, initial_coords, embeddings,
                 length=N_TIMESTEPS, save_interval=SAVE_INTERVAL,
                 friction=FRICTION, masses=MASSES/MASS_SCALE,
                 beta=BETA, dt=DT, save_potential=True,
                 save_forces=True, device=DEVICE,
                 export_interval=EXPORT_INTERVAL,
                 log_interval=LOG_INTERVAL,
                 log_type='write',
                 filename=SAVE_DIR+TAG)
del ref_coords
traj = mysim.simulate()

