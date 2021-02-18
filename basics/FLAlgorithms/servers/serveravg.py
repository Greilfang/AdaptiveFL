import torch
import os

from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

# Implementation for FedAvg Server

class FedAvg(Server):
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, times, selected_rate, packet_loss,threshold):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times, selected_rate)

        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        # packet loss should be passed to set the properties of each user
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            is_eligible = True if i < int(self.selected_rate*total_users) else False
            print("is_eligible:",is_eligible)
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

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_parameters()

            # Evaluate model each interation
            self.evaluate()

            #self.selected_users = self.select_users(glob_iter,self.num_users)
            self.selected_users = self.select_subset_users(glob_iter,self.num_users)
            for user in self.selected_users:
                user.reset_packet_loss()
                user.train(self.local_epochs) #* user.train_samples
            # change the aggregation to the mode with random packet loss
            self.aggregate_parameters_with_packet_loss()
        #print(loss)
        self.save_results()
        self.save_model()