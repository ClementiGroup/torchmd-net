from torch_geometric.data import DataLoader
from torchmdnet.datasets.in_mem_dataset import InMemoryDataset
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import logging
import pickle

def simple_train_loop(model, optimizer, loss_func,
                      train_loader=None,
                      test_loader=None,
                      starting_epoch=0,
                      num_epochs=30,
                      print_freq=10000,
                      model_name="my_model",
                      device=torch.device('cpu')):
    """Simple training loop

    Parameters
    ----------
    model:
        Model to train
    optimizer:
        torch.optim optimizer
    train_loader:
        Dataloader for the training data
    test_loader:
        Dataloader for the testing data
    num_epochs:
        Number of epochs to train
    print_freq:
        Frequency for printing training output to stdoue
    model_name:
        model name
    device:
        torch.device which the model will be moved on and off of
        (between saving model state dicts)
    """


    model.to(device)
    if not train_loader:
        raise RuntimeError("Must include train loader")
    if not test_loader:
        raise RuntimeError("Must include test loader")

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Training Started: {}".format(dt_string))

    epochal_train_losses = []
    epochal_test_losses = []

    for epoch in range(0 + starting_epoch, num_epochs + starting_epoch):
        print("Starting epoch {}".format(epoch))

        #== Train ==#

        model.train()
        optimizer.zero_grad()
        running_train_loss = 0.00

        num_train_batches = 0
        for i, batch_ in enumerate(train_loader):
            batch_.to(device)
            energy, forces = model(batch_.z, batch_.pos, batch=batch_.batch)
            loss = loss_func(batch_.y, forces)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_numpy = loss.detach().cpu().numpy()
            if i%print_freq == 0:
                print("Batch = {}, Train loss:".format(i),loss_numpy)
            running_train_loss += loss_numpy
            num_train_batches += 1

        train_loss = running_train_loss / num_train_batches
        epochal_train_losses.append(train_loss)

        #== Test ==#

        model.eval()
        optimizer.zero_grad()
        running_test_loss = 0.00

        num_test_batches = 0
        for i, batch_ in enumerate(test_loader):
            batch_.to(device)
            energy, forces = model(batch_.z, batch_.pos, batch=batch_.batch)
            loss = loss_func(batch_.y, forces)
            loss_numpy = loss.detach().cpu().numpy()
            if i%print_freq == 0:
                print("Batch = {}, Test loss:".format(i),loss_numpy)
            running_test_loss += loss_numpy
            num_test_batches += 1

        test_loss = running_test_loss / num_test_batches
        epochal_test_losses.append(test_loss)
        print("Epoch {}: Train {} \t Test {}".format(epoch+starting_epoch, train_loss, test_loss))
        logging.info("Epoch {}: Train {} \t Test {}".format(epoch+starting_epoch, train_loss, test_loss))
        print("Saving epoch {} ...".format(epoch))
        with open(model_name+"_state_dict_epoch_{}.pkl".format(epoch), "wb") as modelfile:
            pickle.dump(model.to(torch.device('cpu')).state_dict(), modelfile)
        model.to(device)
    with open(model_name+"_state_dict_epoch_{}.pkl".format(epoch), "wb") as modelfile:
        pickle.dump(model.to(torch.device('cpu')).state_dict(), modelfile)
    np.save("{}_epochal_train_losses.npy".format(model_name), epochal_train_losses)
    np.save("{}_epochal_test_losses.npy".format(model_name), epochal_test_losses)

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Training Finished: {}".format(dt_string))
    
    
def multi_gpu_train_loop(model, optimizer, loss_func,
                      train_loader=None,
                      val_loader=None,
                      starting_epoch=0,
                      num_epochs=30,
                      print_freq=10000,
                      device=torch.device('cpu'),
                      model_name="my_model",
                      save_dir='./'):
    """Simple training loop. Assumes DataParallel object for model



    Parameters
    ----------
    model:
        Model to train
    optimizer:
        torch.optim optimizer
    train_loader:
        Dataloader for the training data
    val_loader:
        Dataloader for the valing data
    num_epochs:
        Number of epochs to train
    print_freq:
        Frequency for printing training output to stdoue
    device:
        torch.device which the model will be moved on and off of
        (between saving model state dicts)
    model_name:
        model name
    save_dir:
        Directory in which results are saved
    """


    model.to(device)
    if not train_loader:
        raise RuntimeError("Must include train loader")
    if not val_loader:
        raise RuntimeError("Must include val loader")

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Training Started: {}".format(dt_string))

    epochal_train_losses = []
    epochal_val_losses = []

    for epoch in range(0 + starting_epoch, num_epochs + starting_epoch):
        print("Starting epoch {}".format(epoch))

        #== Train ==#

        model.train()
        optimizer.zero_grad()
        running_train_loss = 0.00

        num_train_batches = 0
        for i, batch_ in enumerate(train_loader):
            batch_.to(device)
            energy, forces = model(batch_.z, batch_.pos, batch=batch_.batch)
            loss = loss_func(batch_.y, forces)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_numpy = loss.detach().cpu().numpy()
            if i%print_freq == 0:
                print("Batch = {}, Train loss:".format(i),loss_numpy)
            running_train_loss += loss_numpy
            num_train_batches += 1

        train_loss = running_train_loss / num_train_batches
        epochal_train_losses.append(train_loss)

        #== Val ==#

        model.eval()
        optimizer.zero_grad()
        running_val_loss = 0.00

        num_val_batches = 0
        for i, batch_ in enumerate(val_loader):
            batch_.to(device)
            energy, forces = model(batch_.z, batch_.pos, batch=batch_.batch)
            loss = loss_func(batch_.y, forces)
            loss_numpy = loss.detach().cpu().numpy()
            if i%print_freq == 0:
                print("Batch = {}, Test loss:".format(i),loss_numpy)
            running_val_loss += loss_numpy
            num_val_batches += 1

        val_loss = running_val_loss / num_val_batches
        epochal_val_losses.append(val_loss)

        print("Epoch {}: Train {} \t Val {}".format(epoch+starting_epoch, train_loss, val_loss))
        logging.info("Epoch {}: Train {} \t Val {}".format(epoch+starting_epoch, train_loss, val_loss))
        print("Saving epoch {} ...".format(epoch))
        with open(save_dir+model_name+"_state_dict_epoch_{}_val_loss_{}.pkl".format(epoch, val_loss), "wb") as modelfile:
            pickle.dump(model.module.to(torch.device('cpu')).state_dict(), modelfile)
        with open(save_dir+model_name+"_optim_dict_epoch_{}.pkl".format(epoch), "wb") as optimfile:
            pickle.dump(optimizer.state_dict(), optimfile)

        model.to(device)

    with open(save_dir+model_name+"_state_dict_epoch_final.pkl", "wb") as modelfile:
        pickle.dump(model.module.to(torch.device('cpu')).state_dict(), modelfile)
    np.save(save_dir+"{}_epochal_train_losses.npy".format(model_name), epochal_train_losses)
    np.save(save_dir+"{}_epochal_val_losses.npy".format(model_name), epochal_val_losses)

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Training Finished: {}".format(dt_string))
