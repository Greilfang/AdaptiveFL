import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User

# Implementation for FedAvg clients


class UserAVG(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                 local_epochs, optimizer,is_eligible,packet_loss,threshold):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs, is_eligible,packet_loss, threshold)
        
        self.loss = nn.CrossEntropyLoss() if model[1] == "Mclr_CrossEntropy" else nn.NLLLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        print("user type:",type(self.model))

    def reset_optimizer(self,model):
        print("optimizer type:",type(model))
        self.model = copy.deepcopy(model)
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.learning_rate)
        self.relevant_rate = 0
        self.last_server_updates = copy.deepcopy(list(model.parameters()))
        self.last_server_movements = copy.deepcopy(list(model.parameters()))
        self.last_user_updates = copy.deepcopy(list(model.parameters()))
        self.last_user_movements = copy.deepcopy(list(model.parameters()))


    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]
            

    def train(self, epochs, pruning=False, metric='origin'):
        LOSS = 0
        self.model.train()
        if metric == 'cmfl':
            self.clone_model_paramenter(self.model.parameters(), self.last_user_updates)
        elif metric == 'mafl':
            self.clone_model_paramenter(self.model.parameters(), self.last_user_movements)

        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

        if metric == 'cmfl':
            self.calculate_weight_updates()
        elif metric == 'mafl':
            self.calculate_weight_movements()
        self.update_relavance(metric=metric)
        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        return LOSS
    



