import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import numpy as np

class Q_Network(nn.Module):
    """
        Modèle : Réseau de neurones à 2 couches linéaires, sous-classe de nn.Module
            - 1ere couche : input_size données (11) --> hidden_size données
            - Relu
            - 2ème couche : hidden_size données --> output_size (3)
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = functional.relu( self.layer1(x) )
        x = self.layer2(x)
        return x


class QTrainer():
    """
        QTrainer
    """
    def __init__(self, model, lr, gamma):
        self.lr = lr #learning rate
        self.gamma = gamma #Discount rate
        self.model = model  #Model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) #Optimisateur
        self.loss_function = nn.MSELoss() #Loss fonction Mean-Squared error

    def train_update(self, actual_state, li_actions, reward, new_state, game_over) :
        actual_state = torch.tensor(actual_state, dtype=torch.float)
        new_state = torch.tensor(new_state, dtype=torch.float)
        li_actions = torch.tensor(li_actions, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        #Les tenseurs du dessus sont des listes tensor([1,2,3])
        #Mais on veut avoir des tensors à 2 dimensions quitte à n'avoir qu'un seul élément dans une dimension
        #   --> tensor( [ [1,2,3] ] )
        if len(actual_state.shape) == 1:
            actual_state = torch.unsqueeze(actual_state, 0)
            new_state = torch.unsqueeze(new_state, 0)
            li_actions = torch.unsqueeze(li_actions, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over,)

        pred = self.model(actual_state)
        target = pred.clone()

        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(new_state[idx]))

            target[idx][torch.argmax(li_actions[idx]).item()] = Q_new


        self.optimizer.zero_grad()
        loss = self.loss_function(target, pred)
        loss.backward()

        self.optimizer.step()
