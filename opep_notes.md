# mlcg-tools

+ uses main and some notebooks at the moment


# OPEP dataset

+ raw opep dataset `/import/a12/users/nickc/mlcg_opeps/raw`

+ main work folder `/net/data02/nickc/octapeptides`

+ cg coords and forces using opep mapping `/import/a12/users/nickc/opep_cg_mapped_data`


# Torchmd net training workflow

+ repo `https://github.com/ClementiGroup/torchmd-net/tree/main-nec4`

Refer to `/import/a12/users/nickc/example_best_opep_model` for a full model training folder

## The main idea

+ The train and validation data contains for each OPEP need coords / delta forces / embeddings

+ use `examples/train_opep.py` as a template to setup your training

+ run the script from your `new_training` folder (output will go there)


## Build a model from the `.pckl` files

```
PATH = "/import/a12/users/nickc/example_best_opep_model/"
ARGS = PATH+"local_net_20_aa_5_blocks_physnet_res_exclusion_oct_min_args.pkl"
NETWORK_MODEL = PATH+"local_net_20_aa_5_blocks_physnet_res_exclusion_oct_min_state_dict_epoch_8.pkl"

with open(ARGS, "rb") as argfile:
    args = pkl.load(argfile)
with open(NETWORK_MODEL, "rb") as modelfile:
    state_dict = pkl.load(modelfile)
net_model = create_model(args)
net_model.load_state_dict(state_dict)
baseline = BaselineModel(geom_feat, priors, n_beads=len(embeddings))
full_model = CGnet(net_model, baseline)
full_model.to(DEVICE)
```

### Optional

+ get the latest (best priors) training/testting data in a fast storage

```
mkdir opep_train_data
mkdir opep_test_data
mv /net/data02/nickc/oct_test_res_exclusion_oct_min_repul opep_test_data/
mv /net/data02/nickc/oct_train_res_exclusion_oct_min_repul opep_train_data/

```

# Torchmd net simulation workflow

+ use `examples/sim_opep.py` as a template to run simulations

For example run the simulation using
```
examples/sim_opep.py examples/opts.json 2 948
```

where 2 and 948 are the ids of the peptides to run the simulations for.

