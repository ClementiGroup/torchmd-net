# mlcg-tools

+ uses main and some notebooks at the moment


# OPEP dataset

+ raw opep dataset `/import/a12/users/nickc/mlcg_opeps/raw`

+ main work folder `/net/data02/nickc/octapeptides`

+ cg coords and forces using opep mapping `/net/data02/nickc/octapeptides/brooke_map_cg_data/`


# Torchmd net training workflow

+ repo `https://github.com/ClementiGroup/torchmd-net/tree/main-nec4`

Refer to `/import/a12/users/nickc/example_best_opep_model` for a full model training folder

## The main idea

+ The train and validation data contains for each OPEP need coords / delta forces / embeddings

+ use `examples/train_opep.py` as a template to setup your training

+ run the script from your `new_training` folder (output will go there)

### Optional

+ get the data in a fast storage

```
mkdir opep_train_data
mv /net/data02/nickc/oct_test_res_exclusion_oct_min_repul opep_train_data/
mv /net/data02/nickc/oct_train_res_exclusion_oct_min_repul opep_train_data/

```

# Torchmd net simulation workflow

+ use `examples/sim_opep.py` as a template to run simulations

For example run the simulation using
```
examples/sim_opep.py examples/opts.json 2 948
```

where 2 and 948 are the ids of the peptides to run the simulations for.

