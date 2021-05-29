import torch
import os
import numpy as np
import h5py
from utils.model_utils import Metrics
import random
import copy
from functools import cmp_to_key
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import math

def user_cmp(a,b):
    if a.relevant_rate > b.relevant_rate:return -1
    elif a.relevant_rate < b.relevant_rate:return 1
    else:return 0

accuracy_lines = {
    "cmfl":list(),
    "mafl":list(),
    "origin":list()
}
loss_lines = {
    "cmfl":list(),
    "mafl":list(),
    "origin":list()
}
communication_lines = {
    "cmfl":list(),
    "mafl":list(),
    "origin":list()
}

# The component to save the weight of participated users
class WeightSlots:
    def __init__(self,capacity):
        self.capacity = capacity
        self.user_map = dict()
    
    def update_weight(self, mid, weight):
        if len(self.user_map) == self.capacity:
            tid = random.sample(self.user_map.keys(), 1)[0]
            del self.user_map[tid]
        self.user_map[mid] = list(weight)

    def contain(self,mid):
        return True if (mid in self.user_map) else False

    def present(self,mid):
        return self.user_map[mid]



class Server:
    #accuracy_plot = st.empty()
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate ,beta, lamda,
                 num_glob_iters, local_epochs, optimizer,num_users, times, selected_rate):

        global accuracy_lines
        global loss_lines
        global communication_lines
        # Set up the main attributes
        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.users = list()
        self.selected_users = list()
        self.num_users = num_users
        self.beta = beta
        self.lamda = lamda
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc, self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], []
        self.times = times
        self.selected_rate = selected_rate
        self.history_parameters = None
        self.rs_glob_comm = 0

        accuracy_lines = {
            "cmfl":[0 for i in range(self.num_glob_iters)],
            "mafl":[0 for i in range(self.num_glob_iters)],
            "origin":[0 for i in range(self.num_glob_iters)]
        }
        loss_lines = {
            "cmfl":[0 for i in range(self.num_glob_iters)],
            "mafl":[0 for i in range(self.num_glob_iters)],
            "origin":[0 for i in range(self.num_glob_iters)]
        }
        communication_lines = {
            "cmfl":[0 for i in range(self.num_glob_iters)],
            "mafl":[0 for i in range(self.num_glob_iters)],
            "origin":[0 for i in range(self.num_glob_iters)]
        }

        self.accuracy_plot = st.empty()
        self.loss_plot = st.empty()
        self.communication_plot = st.empty()
        self.progress= st.progress(0)


        

        '''Weight slots for history uploaded client model'''
        self.weight_slots = WeightSlots(capacity=num_users)
        # Initialize the server's grads to zeros
        #for param in self.model.parameters():
        #    param.data = torch.zeros_like(param.data)
        #    param.grad = torch.zeros_like(param.data)
        '''Updated weights for two rounds'''
        self.weight_updates = copy.deepcopy(model)
        for param in self.weight_updates.parameters():
            param.data = torch.zeros_like(param.data)
        '''Update movemetns for two rounds'''
        self.weight_movements = copy.deepcopy(model)
        for movement in self.weight_movements.parameters():
            movement.data = torch.zeros_like(movement.data)

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()
        
    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

    def send_weight_updates(self):
        assert (self.selected_users is not None and len(self.selected_users)>0)
        for user in self.selected_users:
            user.set_weight_updates(self.weight_updates)
    
    def send_weight_movements(self):
        assert (self.selected_users is not None and len(self.selected_users)>0)
        for user in self.selected_users:
            user.set_weight_movements(self.weight_movements)



    def send_eligible_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            if user.is_eligible:
                user.set_parameters(self.model)

    def add_parameters_with_packet_loss(self, user, ratio):
        #print(type(user.packet_loss))
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            mask = torch.full(user_param.size(), 1-user.packet_loss)
            bernoulli_mask = torch.bernoulli(mask)
            server_param.data = server_param.data + user_param.data.clone() * bernoulli_mask * ratio / (1-user.packet_loss)
    
    def add_parameters_with_history(self, user, ratio):
        for temp_param, hist_param, server_param, user_param in zip(self.temp_params, self.history_parameters, self.model.parameters(), user.get_parameters()):
            mask = torch.full(user_param.size(), 1-user.packet_loss)
            bernoulli_mask = torch.bernoulli(mask)
            
            server_param.data = server_param.data + user_param.data.clone() * bernoulli_mask * ratio
            temp_param.data = torch.where(temp_param.data==0, (1-bernoulli_mask)*hist_param.data.clone(), temp_param.data)
                
    def add_parameters_with_slots(self, user, ratio):
        chosen_history_parameters = self.weight_slots.present(user.id) if self.weight_slots.contain(user.id) else self.history_parameters
        for temp_param, hist_param, server_param, user_param in zip(self.temp_params,chosen_history_parameters,self.model.parameters(),user.get_parameters()):
            mask = torch.full(user_param.size(),1-user.packet_loss)
            bernoulli_mask = torch.bernoulli(mask)

            server_param.data = server_param.data + user_param.data.clone() * bernoulli_mask * ratio
            temp_param.data = torch.where(temp_param.data==0, (1-bernoulli_mask)*hist_param.data.clone(), temp_param.data)

        if user.packet_loss < 0.02:
            self.weight_slots.update_weight(user.id, user.get_parameters())

   
    def add_parameters(self, user, ratio):
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def calculate_weight_updates(self):
        for weight, hist_param, cur_param in zip(self.weight_updates.parameters(),self.history_parameters,self.model.parameters()):
            weight.data = cur_param.data - hist_param.data

    def calculate_weight_movements(self):
        for mov, hist_param, cur_param in zip(self.weight_movements.parameters(),self.history_parameters,self.model.parameters()):
            mov.data = (cur_param.data - hist_param.data) * cur_param.data

    
    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

    def aggregate_parameters_with_packet_loss(self):
        assert (self.users is not None and len(self.users)>0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters_with_packet_loss(user, user.train_samples / total_train)

    def aggregate_parameters_with_history(self):
        assert (self.users is not None and len(self.users)>0)
        # Do not set parameters to zero first

        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        
        self.history_parameters = copy.deepcopy(list(self.model.parameters()))
        self.temp_params = copy.deepcopy(self.history_parameters)

        for model_param,temp_param in zip(self.model.parameters(),self.temp_params):
            model_param.data = torch.zeros_like(model_param.data)
            temp_param.data = torch.zeros_like(temp_param.data)

        for user in self.selected_users:
            self.add_parameters_with_history(user, user.train_samples / total_train)

        for server_param, temp_param in zip(self.model.parameters(),self.temp_params):
            server_param.data = torch.where(temp_param == 0.0, server_param.data, temp_param.data.clone())
            # server_param.data = server_param.data + temp_param.data.clone()
    
    def aggregate_parameters_with_slots(self, metric = 'origin',n_iter=0):
        '''
        @params:
            cmfl: whether to act communication mitigation federated learning (cmfl)
        '''
        assert (self.users is not None and len(self.users)>0)
        if metric == 'cmfl' or metric == 'mafl' :
            threshold = 0.2 + 1/math.log(n_iter+2) if metric == "cmfl" else 1/math.sqrt(n_iter+1)
            #print("original_len: {}".format(len(self.selected_users)))
            # self.selected_users = list(filter(lambda x: x.relevant_rate > 0.2, self.selected_users))
            #self.selected_users.sort(cmp=None, key=lambda x:x.relevant_rate, reverse=True)
            self.selected_users = sorted(self.selected_users, key = cmp_to_key(user_cmp))

            anchor = 0
            for user in self.selected_users:
                if user.relevant_rate > threshold:
                    anchor = anchor + 1


            print("threshold:",threshold,"anchor:",anchor)
            anchor = -1 if anchor == self.num_users else anchor
            self.selected_users=self.selected_users[anchor:]
            print("threshold:",threshold,"anchor:",anchor)
            print([user.relevant_rate for user in self.selected_users])


        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples

        # history parameters saves the global model in the previous round
        self.history_parameters = copy.deepcopy(list(self.model.parameters()))
        self.temp_params = copy.deepcopy(self.history_parameters)

        # clear temp_params before each aggregation
        for model_param, temp_param in zip(self.model.parameters(),self.temp_params):
            model_param.data = torch.zeros_like(model_param.data)
            temp_param.data = torch.zeros_like(temp_param.data)

        for user in self.selected_users:
            self.add_parameters_with_slots(user, user.train_samples / total_train)
        
        for server_param, temp_param in zip(self.model.parameters(), self.temp_params):
            server_param.data = torch.where(temp_param == 0.0, server_param.data, temp_param.data.clone())
    
    def global_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        
        total_train = 0
        for user in self.users:
            total_train += user.train_samples

        for user in self.users:
            self.add_parameters(user, user.train_samples / total_train)

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta) * pre_param.data + self.beta * param.data

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    
    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        #np.random.seed(round)
        return np.random.choice(self.users, num_users, replace=False) #, p=pk)

    # 0.72 is the default settadd_parameters_with_droprate clients from ratio * total_users 
    def select_subset_users(self,round,num_users):
        '''selects num_clients clients from ratio * total_users 
        Args:
            ratio: decide the size of subset of users that may be selected from
        
        Return:
            list of selected clients objects
        '''
        if num_users >= len(self.users)*self.selected_rate:
            print("All users are selected")
        # if self.selected_rate == 1:
        #     print("All users may be selected")
        num_users = min(num_users,len(self.users))
        eid = int(len(self.users)*self.selected_rate)
        np.random.seed(round)
        return np.random.choice(self.users[:eid],num_users,replace=False)

    def select_candidate_users(self,round,num_users):
        #print(self.selected_rate)
        '''selects num_clients clients from ratio * total_users 
        Args:
            ratio: decide the size of subset of users that may be selected from
        
        Return:
            list of selected clients objects
        '''
        candidate_users = num_users
        if candidate_users >= len(self.users)*self.selected_rate:
            print("All users are candidates")
        # if self.selected_rate == 1:
        #     print("All users may be selected")
        candidate_users = min(candidate_users,len(self.users))
        eid = int(len(self.users)*self.selected_rate)
        np.random.seed(round)
        return np.random.choice(self.users[:eid],candidate_users,replace=False)
    

    # define function for persionalized agegatation.
    def persionalized_update_parameters(self,user, ratio):
        # only argegate the local_weight_update
        for server_param, user_param in zip(self.model.parameters(), user.local_weight_updated):
            server_param.data = server_param.data + user_param.data.clone() * ratio


    def personalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            #self.add_parameters(user, 1 / len(self.selected_users))

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
    
    def personalized_aggregate_parameters_with_packet_loss(self, packet_loss=0.0):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        for user in self.selected_users:
            self.add_parameters_with_packet_loss(user, user.train_samples / total_train, packet_loss)
            #self.add_parameters(user, 1 / len(self.selected_users))

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
            
    # def save_results(self):
    #     alg = self.dataset + "_" + self.algorithm
    #     alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_"+str(self.selected_rate)+"r"+"_" + str(self.local_epochs)
        
    #     if(self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
    #         alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
    #     alg = alg + "_" + str(self.times)
    #     if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
    #         with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
    #             hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
    #             hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
    #             hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
    #             hf.close()
        
    #     # store persionalized value
    #     alg = self.dataset + "_" + self.algorithm + "_p"
    #     alg = alg  + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b"+ "_" +str(self.selected_rate)+"r"+"_"+ str(self.local_epochs)
    #     alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
    #     alg = alg + "_" + str(self.times)
    #     if (len(self.rs_glob_acc_per) != 0 &  len(self.rs_train_acc_per) & len(self.rs_train_loss_per)) :
    #         with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
    #             hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc_per)
    #             hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
    #             hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
    #             hf.close()

    def save_results(self):
        alg = self.dataset + "_" + self.algorithm
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_"+str(self.selected_rate)+"r"+"_" + str(self.local_epochs)
        if(self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.close()
        
        # store persionalized value
        alg = self.dataset + "_" + self.algorithm + "_p"
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_"+str(self.selected_rate)+"r"+"_" + str(self.local_epochs)
        if(self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc_per) != 0 &  len(self.rs_train_acc_per) & len(self.rs_train_loss_per)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
                hf.close()


    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = list()
        tot_correct = list()
        losses = []
        for c in self.users:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss(self):
        num_samples = list()
        tot_correct = list()
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def test_persionalized_model(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        for c in self.users:
            ct, ns = c.test_persionalized_model()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss_persionalized_model(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_persionalized_model() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def reset_model(self,model):
        self.model = copy.deepcopy(model)
        self.rs_glob_acc = list()
        self.rs_train_acc = list()
        self.rs_train_loss = list()
        self.rs_glob_comm = 0
        self.weight_slots = WeightSlots(capacity=self.num_users)
        for user in self.users:
            user.reset_optimizer(model)
        # Initialize the server's grads to zeros
        #for param in self.model.parameters():
        #    param.data = torch.zeros_like(param.data)
        #    param.grad = torch.zeros_like(param.data)
        '''Updated weights for two rounds'''
        self.weight_updates = copy.deepcopy(model)
        for param in self.weight_updates.parameters():
            param.data = torch.zeros_like(param.data)
        '''Update movemetns for two rounds'''
        self.weight_movements = copy.deepcopy(model)
        for movement in self.weight_movements.parameters():
            movement.data = torch.zeros_like(movement.data)

    def evaluate(self,glob_iter,metric):
        # accuracy_line = st.line_chart([])
        stats = self.test()  
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        self.rs_glob_comm = self.rs_glob_comm + len(self.selected_users)
        #print("stats_train[1]",stats_train[3][0])
        if metric == "cmfl":
            accuracy_lines["cmfl"][glob_iter]=glob_acc
            loss_lines['cmfl'][glob_iter]=train_loss
            communication_lines['cmfl'][glob_iter] = self.rs_glob_comm
        elif metric == "mafl":
            accuracy_lines["mafl"][glob_iter]=glob_acc
            loss_lines['mafl'][glob_iter]=train_loss
            communication_lines['mafl'][glob_iter] = self.rs_glob_comm
        else:
            accuracy_lines["origin"][glob_iter]=glob_acc
            loss_lines['origin'][glob_iter]=train_loss
            communication_lines['origin'][glob_iter] = self.rs_glob_comm

        self.accuracy_plot.line_chart(accuracy_lines)  
        self.loss_plot.line_chart(loss_lines)  
        self.communication_plot.line_chart(communication_lines)
        
        print("Average Global Accurancy: ", glob_acc)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ",train_loss)

    def evaluate_personalized_model(self):
        stats = self.test_persionalized_model()  
        stats_train = self.train_error_and_loss_persionalized_model()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ",train_loss)

    def evaluate_one_step(self):
        for c in self.users:
            c.train_one_step()

        stats = self.test()  
        stats_train = self.train_error_and_loss()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.h)

        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ",train_loss)
