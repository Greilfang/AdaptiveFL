import torch
import os

from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data

from csvec import CSVec
import streamlit as st
import numpy as np

# Implementation for FedAvg Server

class FedAvg(Server):
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, times, selected_rate, packet_loss,threshold,sketched):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times, selected_rate)

        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])

        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            is_eligible = True if i < int(self.selected_rate*total_users) else False
            # print("is_eligible:", id)
            user = UserAVG(device, id, train, test, model, batch_size, learning_rate,beta,lamda, local_epochs, optimizer,is_eligible, packet_loss, threshold)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    # baseline for fedavg
    def origin_train(self):
        loss = list()
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            self.progress.progress((glob_iter + 1)/(self.num_glob_iters + 1))
            #loss_ = 0
            # set self.local_model and self.model 
            self.send_parameters()

            # Evaluate model each iteration
            self.evaluate(glob_iter,metric="origin")

            self.selected_users = self.select_subset_users(glob_iter,self.num_users)
            # send updates to users
            self.send_weight_updates()
            for user in self.selected_users:
                user.reset_packet_loss()
                user.train(epochs=self.local_epochs, pruning=False, metric='origin') #* user.train_samples
            # change the aggregation to the mode with random packet loss
            # self.aggregate_parameters_with_packet_loss()
            # self.aggregate_parameters_with_history()
            self.aggregate_parameters_with_slots(metric="origin", n_iter = glob_iter)

        # self.save_results()
        # self.save_model()
    
    def mafl_train(self):
        print("mafl train !!!")
        loss = list()
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter,"-------------")
            self.progress.progress((glob_iter + 1)/(self.num_glob_iters + 1))
            self.send_parameters()
            self.evaluate(glob_iter,metric='mafl')
            self.selected_users = self.select_candidate_users(glob_iter,self.num_users)
            # send movements to users
            self.send_weight_movements()
            for user in self.selected_users:
                user.reset_packet_loss()
                user.train(epochs=self.local_epochs, pruning=False, metric='mafl')
                user.update_relavance(metric='mafl')
            
            self.aggregate_parameters_with_slots(metric = 'mafl', n_iter = glob_iter)
            self.calculate_weight_movements()
        # self.save_results()
        # self.save_model()




    def cmfl_train(self):
        loss = list()
        # The num of round t
        # o upload unimportant weight
        extended_round = 10
        for glob_iter in range(self.num_glob_iters):
            print("------------Round Number: ",glob_iter, " --------------")
            self.progress.progress((glob_iter + 1)/(self.num_glob_iters + 1))
            # set self.local_model and self.model 
            self.send_parameters()
            self.evaluate(glob_iter,metric='cmfl')

            #print("self.num_users:",self.num_users)
            self.selected_users = self.select_candidate_users(glob_iter, self.num_users)
            self.send_weight_updates()
            #print("server :\n {} ".format(list(self.weight_updates.parameters())[0]))
            for user in self.selected_users:
                # To act packet-loss-aware pruning
                user.reset_packet_loss()
                # training settings
                user.train(epochs=self.local_epochs, pruning=False, metric='cmfl')
                # the server updates the similarity record of local and global updates
                user.update_relavance(metric='cmfl')
                #print("user {}:{}".format(user.id,user.relevant_rate))

            self.aggregate_parameters_with_slots(metric = 'cmfl', n_iter = glob_iter)
            self.calculate_weight_updates()

        # self.save_results()
        # self.save_model()
