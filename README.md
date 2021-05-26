# torchmd-net

## Installation
Create a new conda environment using Python 3.8 via
```
conda create --name torchmd python=3.8
conda activate torchmd
```

Then, install PyTorch according to your hardware specifications (more information [here](https://pytorch.org/get-started/locally/#start-locally)), e.g. for CUDA 11.1 and the most recent version of PyTorch use
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```

Download and install the `torchmd-net` repository via
```
git clone https://github.com/compsciencelab/torchmd-net.git
pip install -e torchmd-net/
```

Finally, install `torch-geometric` with its dependencies as it is specified [here](https://github.com/rusty1s/pytorch_geometric#installation). Example for PyTorch 1.8 and CUDA 11.1:
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-geometric
```

## Usage
Specifying training arguments can either be done via a configuration yaml file or through command line arguments directly. An example configuration file for a TorchMD Graph Network can be found at [examples/graph-network.yaml](https://github.com/compsciencelab/torchmd-net/blob/main/examples/graph-network.yaml). For an example on how to train the network on the QM9 dataset, see [examples/train_GN_QM9.sh](https://github.com/compsciencelab/torchmd-net/blob/main/examples/train_GN_QM9.sh). GPUs can be selected by their index by listing the device IDs (coming from `nvidia-smi`) in the `CUDA_VISIBLE_DEVICES` environment variable. Otherwise, the argument `--ngpus` can be used to select the number of GPUs to train on (-1 uses all available GPUs or the ones specified in `CUDA_VISIBLE_DEVICES`).
```
mkdir output
CUDA_VISIBLE_DEVICES=0 python torchmd-net/scripts/torchmd_train.py --conf torchmd-net/examples/graph-network.yaml --dataset QM9 --log-dir output/
```

## Creating a new dataset
If you want to train on custom data, first have a look at `torchmdnet.datasets.Custom`, which provides functionalities for 
loading a NumPy dataset consisting of atom types and coordinates, as well as energies, forces or both as the labels.
Alternatively, you can implement a custom class according to the torch-geometric way of implementing a dataset. That is, 
derive the `Dataset` or `InMemoryDataset` class and implement the necessary functions (more info [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-your-own-datasets)). The dataset must return torch-geometric `Data` 
objects, containing at least the keys `z` (atom types) and `pos` (atomic coordinates), as well as `y` (label), `dy` (derivative of the label w.r.t atom coordinates) or both.

### Custom prior models
In addition to implementing a custom dataset class, it is also possible to add a custom prior model to the model. This can be
done by implementing a new prior model class in `torchmdnet.priors` and adding the argument `--prior-model <PriorModelName>`.
As an example, have a look at `torchmdnet.priors.Atomref`.

## Multi-Node Training
__Currently does not work with the most recent PyTorch Lightning version. Tested up to pytorch-lightning==1.2.10__

In order to train models on multiple nodes some environment variables have to be set, which provide all necessary information to PyTorch Lightning. In the following we provide an example bash script to start training on two machines with two GPUs each. The script has to be started once on each node. Once [`train.py`](https://github.com/compsciencelab/torchmd-net/blob/main/scripts/train.py) is started on all nodes, a network connection between the nodes will be established using NCCL.

In addition to the environment variables the argument `--num-nodes` has to be specified with the number of nodes involved during training.

```
export NODE_RANK=0
export MASTER_ADDR=hostname1
export MASTER_PORT=12910

mkdir -p output
CUDA_VISIBLE_DEVICES=0,1 python torchmd-net/scripts/train.py --conf torchmd-net/examples/graph-network.yaml --num-nodes 2 --log-dir output/
```

- `NODE_RANK` : Integer indicating the node index. Must be `0` for the main node and incremented by one for each additional node.
- `MASTER_ADDR` : Hostname or IP address of the main node. The same for all involved nodes.
- `MASTER_PORT` : A free network port for communication between nodes. PyTorch Lightning suggests port `12910` as a default.

### Known Limitations
- Due to the way PyTorch Lightning calculates the number of required DDP processes, all nodes must use the same number of GPUs. Otherwise training will not start or crash.
- We observe a 50x decrease in performance when mixing nodes with different GPU architectures (tested with RTX 2080 Ti and RTX 3090).
