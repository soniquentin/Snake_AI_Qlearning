# Snake AI (Machine Learning)


## Introduction

Petit projet qui permet de visualiser un agent qui s'entraine à snake et qui s'améliore au fil des parties. A été utilisé un algorithme de Q-Learning.

## Lancer l'entrainement

Veillez bien à lancer le fichier texte `requirement.txt` afin d'installer les bonnes versions des packages utilisés. Pour ce faire, utilisez la commande :
```pip3 install -r requirements.txt```

Pour lancer l'entrainement, il suffit de lancer le programme Python `agent.py`

## Pendant l'entrainement

Pendant l'entrainement, vous verrez l'agent apprendre à jouer. 3 valeurs sont affichées en haut à gauche de l'écran :
* Générations : c'est le nombre de parties qui a été jouées en tout.
* Score : Nombre de pommes mangées par le snake durant la partie en cours.
* Vitesse : vitesse d'entrainement. Par défaut, la vitesse est au max pour aller entrainer au plus vite. Mais si vous voulez ralentir pour mieux apprécier les déplacements de l'agent, vous pouvez utiliser les flèches directionnels de votre clavier.


## Quelques remarques

J'ai mis le paramètre `epsilon` à 80. Ce paramètre est un ordre de grandeur du nombre de parties où l'algorithme va jouer un peu aléatoirement pour explorer des possibilités. Ici, cela signifie que l'agent va jouer presque totalement aléatoirement pendant les 80 premières parties.
