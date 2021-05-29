#!/usr/bin/env python
from re import S
import h5py
import matplotlib.pyplot as plt
import argparse
import importlib
import random
import os

from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverpFedMe import pFedMe
from FLAlgorithms.servers.serverperavg import PerAvg
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *

import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd


import torch
torch.manual_seed(0)

def main(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, gpu, selected_rate,packet_loss, threshold, sketched):

    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    real_model = None 

    for i in range(times):
        print("---------------Running time:-------------",i)
        # Generate model
        if(model == "mclr"):
            if(dataset == "Mnist"):
                real_model = Mclr_Logistic().to(device), model
            else:
                real_model = Mclr_Logistic(60,10).to(device), model
                
        if(model == "cnn"):
            if(dataset == "Mnist"):
                real_model = Net().to(device), model
            elif(dataset == "Cifar10"):
                real_model = CifarNet().to(device), model
            
        if(model == "dnn"):
            if(dataset == "Mnist"):
                real_model = DNN().to(device), model
            else: 
                real_model = DNN(60,20,10).to(device), model

        # select algorithm

        server = FedAvg(device, dataset, algorithm, real_model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i, selected_rate, packet_loss, threshold, sketched)
        
        #print(algorithm)
        for alg in server.algorithm:
            server.reset_model(real_model[0])
            #server = FedAvg(device, dataset, alg, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i, selected_rate, packet_loss, threshold, sketched)
            if alg == "CMFL_FedAvg":
                server.cmfl_train()
            elif alg == "MAFL_FedAvg":
                server.mafl_train()
            elif alg == "Original_FedAvg":
                server.origin_train()
        #server.test()


def task_selector():
    task_container = st.sidebar.beta_expander("Configure a Task", True)
    with task_container:
        selected_dataset = st.selectbox("Choose a dataset", ("Synthetic","Cifar10", "Mnist"),help="3 datasets")
        selected_model = st.selectbox(
            'Which model?',
        ['dnn','mclr','cnn'])

        selected_algorithm = st.multiselect(
            'Which algorithm?',
        ['CMFL_FedAvg','MAFL_FedAvg','Original_FedAvg']
        )

    return selected_dataset, selected_model, selected_algorithm

def parameter_selector(dataset):
    parameter_container= st.sidebar.beta_expander("Configure training parameters", True)
    with parameter_container:
        selected_batch_size = st.slider(label="batch size", min_value=1, max_value=64,step=1,value=32)
        selected_lr = st.slider('learning rate (× 0.001)',min_value=1,max_value=100,value=5,step=1)*0.001
        selected_global_iter = st.number_input('global iterations',min_value=1, step=1,value=200)
        selected_local_epoch = st.number_input('local epochs',min_value=1, max_value=50, step=1, value=5)

        selected_num_user_per_round = None
        if dataset in ("Cifar10","Mnist"):
            selected_num_user_per_round = st.number_input('number of users', min_value=1, max_value=10, step=1, value=10, help="For Cifar10 and Mnist, the total num of users is 20.")
        else:
            selected_num_user_per_round = st.number_input('number of users', min_value=1, max_value=50, step=1, value=20, help="For Synthetic, the total num of users is 100")


    return selected_batch_size,selected_lr,selected_global_iter,selected_local_epoch,selected_num_user_per_round

def experiment_selector():
    experiment_container= st.sidebar.beta_expander("Configure training parameters", True)
    with experiment_container:
        selected_eligible_rate = st.number_input('user eligible rate',min_value=0.7, max_value=1.0,value=1.0,step=0.1)
        selected_threshold = st.number_input(label="latency threshold", min_value=0, max_value=2000,value=400,step=1)
    
    return selected_eligible_rate,selected_threshold

if __name__ == "__main__":
    st.title("Loss Tolerant Federated Learning")
    selected_dataset, selected_model, selected_algorithm=task_selector()
    selected_batch_size,selected_lr,selected_global_iter,selected_local_epoch,selected_num_user_per_round = parameter_selector(selected_dataset)
    selected_eligible_rate,selected_threshold = experiment_selector()
    # accuracy_line = st.line_chart([])

    parser = argparse.ArgumentParser()

    # expander = st.beta_expander("FAQ")
    # expander.write("Here you could put in some really, really long explanations...")
    
    # Add the options into the parser
    parser.add_argument("--dataset", type=str, default = selected_dataset, choices=["Mnist", "Synthetic", "Cifar10"])
    parser.add_argument("--model", type=str, default = selected_model, choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--batch_size", type=int, default=selected_batch_size)
    parser.add_argument("--learning_rate", type=float, default=selected_lr, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=selected_global_iter)
    parser.add_argument("--local_epochs", type=int, default=selected_local_epoch)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=list, default=selected_algorithm) 
    parser.add_argument("--numusers", type=int, default=selected_num_user_per_round, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.09, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--gpu", type=int, default=-1, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    parser.add_argument("--selected_rate",type=float, default=selected_eligible_rate, help="The proption of clients that meet the latency requirement" )
    parser.add_argument("--packet_loss",type=str,default='adaptive', help="the packet loss rate")
    parser.add_argument("--threshold",default=selected_threshold,help="The threshold to calculate packet loss")
    parser.add_argument("--sketched",default=False, help="Whether use sketch to help compress gradients")
    args = parser.parse_args()

    # print("=" * 80)
    # print("Summary of training process:")
    # print("Algorithm: {}".format(args.algorithm))
    # print("Batch size: {}".format(args.batch_size))
    # print("Learing rate       : {}".format(args.learning_rate))
    # print("Average Moving       : {}".format(args.beta))
    # print("Subset of users      : {}".format(args.numusers))
    # print("Number of global rounds       : {}".format(args.num_global_iters))
    # print("Number of local rounds       : {}".format(args.local_epochs))
    # print("Dataset       : {}".format(args.dataset))
    # print("Local Model       : {}".format(args.model))
    # print("Selection Rate       : {}".format(args.selected_rate))
    # print("Packet Loss       : {}".format(args.packet_loss))
    # print("Latency Threshold       : {}".format(args.threshold))
    # print("Count Sketch       : {}".format(args.sketched))
    # print("=" * 80)

    left_column, right_column = st.sidebar.beta_columns(2)
    pressed = left_column.button('Start Train')
    if pressed:
        right_column.write("Woohoo!")
        main(
            dataset=args.dataset,
            algorithm = args.algorithm,
            model=args.model,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            beta = args.beta, 
            lamda = args.lamda,
            num_glob_iters=args.num_global_iters,
            local_epochs=args.local_epochs,
            optimizer= args.optimizer,
            numusers = args.numusers,
            K=args.K,
            personal_learning_rate=args.personal_learning_rate,
            times = args.times,
            gpu=args.gpu,
            selected_rate=args.selected_rate,
            packet_loss = args.packet_loss,
            threshold = args.threshold,
            sketched = args.sketched
            )
