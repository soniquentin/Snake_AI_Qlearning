import random
from collections import deque
from Snake_Board_Class import Snake, Board
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from model import Q_Network, QTrainer
from tools import plot, HiddenPrints
import pygame as pyg


MAX_MEMORY = 100000
BATCH_SIZE = 1000




class Agent():
    """
        Agent qui va s'entrainer
    """
    def __init__(self, LR, gamma, epsilon, size = 40, coef_mult = 15):
        self.lr = LR
        self.gamma = gamma
        self.epsilon = epsilon #Nombre de partis avant que l'aspect aléatoire des prises de décisions disparaisse
        self.size = size
        self.coef_mult = coef_mult #Taille des cases
        self.memory = deque(maxlen=MAX_MEMORY) #Popleft à gauche quand memory est de taille plus grande que MAX_MEMORY
        self.generations = 0
        self.model = Q_Network(11,256,3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=gamma)
        self.time_tick = 0.01
        self.timesleep = 0
        self.last_scores = []
        self.mean_scores = []


    def update_plot(self, plateau):
        nb_means_stored = len(self.last_scores)
        if nb_means_stored == 30 :
            self.last_scores.pop(0)
        self.last_scores.append(plateau.score)
        nb_means_stored = len(self.last_scores)
        avg = 0
        for i in range(nb_means_stored) :
            avg += self.last_scores[i]
        self.mean_scores.append(avg/nb_means_stored)
        plot(self.mean_scores)


    def play_one_game(self, sur):

        game_over = False

        plateau = Board(self.size, self.lr, self.gamma, self.epsilon, self.generations, self.model, self.trainer)

        while not game_over :

            for event in pyg.event.get():
                #Pour controler la vitesse :
                if event.type == pyg.KEYDOWN:
                    if event.key == 1073741905 :
                        self.timesleep += self.time_tick
                    elif event.key == 1073741906 and self.timesleep != 0 :
                        self.timesleep -= self.time_tick
                        if self.timesleep < self.time_tick/2 :
                            self.timesleep = 0
            if self.timesleep > 0 :
                time.sleep(self.timesleep)


            actual_state = plateau.get_state()
            li_actions = plateau.get_action()

            action = np.argmax(li_actions)
            x_dir, y_dir = plateau.snake.direction
            if action == 2 : #On tourne à droite
                action = (-y_dir , x_dir)
            elif action == 1 : #On tourne à gauche
                action = (y_dir , -x_dir)
            else : #Ne fait rien
                action = None

            game_over, reward = plateau.update(action)

            new_state = plateau.get_state()

            #Train en short memory
            plateau.trainer.train_update(actual_state, li_actions, reward, new_state, game_over)
            self.memory.append((actual_state, li_actions, reward, new_state, game_over))

            plateau.draw(sur, self.coef_mult) #On dessine le plateau

            #Affichage du texte
            font = pyg.font.Font('freesansbold.ttf', 20)
            score_text = font.render('Score : {}'.format(plateau.score), True, (255, 255, 255))
            score_textRect = score_text.get_rect()
            if self.timesleep == 0 :
                vitesse = 'MAX'
            else :
                vitesse = round(1/self.timesleep , 1)
            speed_text = font.render('Vitesse : {}'.format(vitesse), True, (255, 255, 255))
            generation_text = font.render('Générations : {}'.format(self.generations), True, (255, 255, 255))
            sur.blit(generation_text, (10,0))
            sur.blit(score_text, (10,20))
            sur.blit(speed_text, (10,40))


            pyg.display.flip()



        #Train en long memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # On prend un échantillon aléatoire de taille BATCH_SIZE de self.memory
        else:
            mini_sample = self.memory

        actual_states, actions, rewards, new_states, game_overs = zip(*mini_sample)
        plateau.trainer.train_update(actual_states, actions, rewards, new_states, game_overs)

        #On trace les courbes
        with HiddenPrints():
            self.update_plot(plateau)


    def train(self):

        pyg.init()

        sur = pyg.display.set_mode( (self.coef_mult*self.size , self.coef_mult*self.size) )
        pyg.display.set_caption('Snake')
        sur.fill((255, 255, 255))


        while True :
            self.generations += 1
            sur.fill((255, 255, 255))

            self.play_one_game(sur)





if __name__ == "__main__" :
    new_agent = Agent(LR = 0.001, gamma = 0.9, epsilon = 80, size = 25, coef_mult = 25)
    new_agent.train()
