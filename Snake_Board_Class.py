import pygame as pyg
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from model import Q_Network, QTrainer
from tools import decreasing_exp


class Snake():
    """
        Snake
    """
    def __init__(self, initial_body_pos):
        x,y = initial_body_pos
        self.body = [(x,y), (x,y-1), (x,y-2) ]
        self.direction = (0,1) #(0,1) : bas, (1,0) : droite, (0,-1) : haut, (-1,0) : gauche
        self.body_size = len(self.body)


    def update(self, action,apple_pos) : #Met à jour le snake

        #Action est une paire de {0,1}^2
        x_head, y_head = self.body[0]

        if action != None :
            self.direction = action


        vel_x, vel_y = self.direction

        next_box = x_head + vel_x, y_head + vel_y #Prochaine position de la tête

        apple_touched = (next_box == apple_pos) #Vaut TRUE si la tête du snake a touché la pomme

        new_value = next_box
        for i in range(self.body_size): #On va décaler tout le corps
            old_value = self.body[i]
            self.body[i] = new_value
            new_value = old_value

        free_box = new_value
        if apple_touched :
            self.body.append(new_value) #On réappend la précédente queue normalement supprimée
            free_box = None
            self.body_size += 1

        return apple_touched, next_box, free_box


class Board():
    """
        Game Board, c'est ce qui va nous servir d'agent aussi
    """
    def __init__(self, size, LR, gamma,epsilon, generations, model, trainer):
        self.size = size #Taille du plateau de jeu (format carré)
        self.snake = Snake( (size//2, size//2) )
        self.remaining_cases = [(i,j) for i in range(size) for j in range(size)]
        for x,y in self.snake.body :
            self.remaining_cases.remove((x,y))
        self.epsilon = epsilon
        self.score = 0
        self.snake_hungry = 0 #Compte le nombre de déplacement sans que le snake ait mangé de pomme
        self.snake_hungry_limit = size*size + 1 #Nombre de déplacements à partir duquel le snake meurt car n'a pas mangé de pomme
        self.generations = generations #Numéro de la partie en cours
        self.apple_pos = self.generate_apple() #Position de la pomme, elle ne peut pas être sur une case du corps du snake
        self.model = model
        self.trainer = trainer


    def update(self, action) : #Met à jour le plateau à chaque frame
        reward = 0
        game_over = False
        step_forward = False #step_forward = True quand il y a eu une update du snake


        step_forward = True
        apple_touched, new_box, free_box = self.snake.update(action, self.apple_pos)

        #On met à jour les cases libres qui peuvent héberger une pomme
        try :
            self.remaining_cases.remove(new_box)
        except Exception :
            reward = -10
            game_over = True

        if self.snake_hungry == self.snake_hungry_limit :
            reward = -10
            game_over = True

        if free_box != None :
            self.remaining_cases.append(free_box)

        if apple_touched : #change_pomme vaut True si la tête du snake a rencontré la pomme
            self.apple_pos = self.generate_apple()
            reward = 10
            self.snake_hungry = 0
            self.score += 1
        else :
            self.snake_hungry += 1

        return game_over, reward


    def generate_apple(self) :
        """
            Génère une position aléatoire de la pomme
        """
        new_apple_index = random.randint(0, len(self.remaining_cases) - 1) #Tire un index au hasard parmis les case restante
        return self.remaining_cases[new_apple_index]


    def draw(self, sur, coef_mult):
        import pygame as pyg

        for i in range(self.size) :
            pyg.draw.line(sur, color = (0,0,0), start_pos = (0, i*coef_mult), end_pos = (self.size*coef_mult, i*coef_mult))
        for j in range(self.size) :
            pyg.draw.line(sur, color = (0,0,0), start_pos = (j*coef_mult, 0), end_pos = (j*coef_mult,self.size*coef_mult))

        #Dessin des carrés
        for i,j in self.snake.body :
            rect = pyg.Rect( coef_mult*i + 1 , coef_mult*j + 1, coef_mult-1,coef_mult-1 )
            pyg.Surface.fill(sur, color = (230,230,230),  rect = rect )
        for i,j in self.remaining_cases :
            rect = pyg.Rect( coef_mult*i + 1 , coef_mult*j + 1, coef_mult-1,coef_mult-1 )
            pyg.Surface.fill(sur, color = (70,70,70),  rect = rect )
        x_apple, y_apple = self.apple_pos
        rect = pyg.Rect( coef_mult*x_apple + 1 , coef_mult*y_apple + 1, coef_mult-1,coef_mult-1 )
        pyg.Surface.fill(sur, color = (200,0,0),  rect = rect )


    def get_state(self):
        li_states = []

        x_head, y_head = self.snake.body[0]
        x_apple, y_apple = self.apple_pos
        snake_direction = self.snake.direction

        down = ( snake_direction == (0,1) ) #Le snake va vers le bas
        right = (snake_direction == (1,0)) #Le snake va vers la droite
        up = (snake_direction == (0,-1)) #Le snake va vers le haut
        left = (snake_direction == (-1,0)) #Le snake va vers la gauche

        up_case = ( (x_head, y_head-1) not in self.remaining_cases) #Regarde si la case du haut est libre
        down_case = ( (x_head, y_head+1) not in self.remaining_cases)
        right_case = ( (x_head+1, y_head) not in self.remaining_cases)
        left_case = ( (x_head-1, y_head) not in self.remaining_cases)

        #True si y'a un mur à droite ou le corps du snake
        li_states.append( (right and down_case )
                        or (down and left_case)
                        or (up and right_case)
                        or (left and up_case ) )
        #True si y'a un mur à gauche ou le corps du snake
        li_states.append( (right and up_case )
                        or (down and right_case)
                        or (up and left_case)
                        or (left and down_case ) )
        #True si y'a un mur devant ou le corps du snake
        li_states.append( (right and right_case )
                        or (down and down_case)
                        or (up and up_case)
                        or (left and left_case ) )

        li_states.append(down)
        li_states.append(right)
        li_states.append(up)
        li_states.append(left)

        li_states.append(x_head < x_apple) #La pomme est à droite du snake
        li_states.append(x_head > x_apple)  #La pomme est à gauche du snake
        li_states.append(y_head < y_apple)  #La pomme est en-dessous du snake
        li_states.append(y_head > y_apple) #La pomme est au-dessus du snake

        return np.array(li_states, dtype=int) #Liste à 11 éléments

    def get_action(self):
        """
            Dans les premières parties, les actions seront aléatoires et au fur et à mesure,
            get_action va s'aider du modèle pour prendre ses décisions
        """
        state = self.get_state()

        li_actions = [0,0,0] #L'action choisit sera la seule action à 1.
        #1er élément : ne rien faire
        #2eme élément : tourner à droite
        #3ème élément : tourner à gauche

        randomness = random.random()
        if randomness <= decreasing_exp(self.epsilon,self.generations) :
            action_choisie = random.randint(0,2)
        else :
            state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state)
            action_choisie = torch.argmax(prediction).item()

        li_actions[action_choisie] = 1

        return np.array(li_actions, dtype = int)
