Comparaison de Stratégies d'Optimisation Distribuée pour Vision Transformer

Ce projet a pour objectif d'évaluer et de comparer l'efficacité de deux stratégies de formation distribuée — All-Reduce et Parameter Server — par rapport à une baseline sur un seul GPU, pour l'entraînement d'un modèle Vision Transformer (ViT).
Table des Matières

    Objectif du Projet

    Architectures Étudiées

    Structure du Projet

    Mise en Route

    Comment Lancer les Entraînements

    Critères d'Évaluation

Objectif du Projet

L'objectif principal est de mesurer et d'analyser les performances de différentes architectures de communication pour la synchronisation des gradients lors de l'entraînement d'un grand modèle de vision.
Architectures Étudiées
Baseline (Single GPU)

Il s'agit de l'entraînement standard sur une seule carte graphique. C'est la référence par rapport à laquelle les performances des autres stratégies seront mesurées.

    Implémentation : training/single_gpu_main.py.

All-Reduce (Décentralisé)

Chaque processus (GPU) communique ses gradients à tous les autres. Il n'y a pas de serveur central. Les gradients sont moyennés de manière décentralisée (ring-allreduce).

    Implémentation : training/all_reduce_main.py.

Parameter Server (Centralisé)

Un processus (le Parameter Server) est seul responsable de la mise à jour des poids. Les autres processus (les workers) envoient leurs gradients au serveur.

    Implémentation : training/parameter_server_main.py.

Structure du Projet
```
Distributed-ViT-Project/
│
├── Data/
│   ├── prepare_data.py
│   └── tiny-imagenet-200/
│
├── training/
│   ├── single_gpu_main.py
│   ├── all_reduce_main.py
│   └── parameter_server_main.py
│
├── requirements.txt
├── README.md
└── LICENSE
```
Mise en Route
Prérequis

    Python 3.8+

    PyTorch 1.9+

    CUDA 11.0+ & NCCL

    Horovod

Installation

    Clonez le dépôt :

    git clone https://github.com/cchahid/Distributed-ViT-Project.git
    cd Distributed-ViT-Project

    Installez les dépendances :

    pip install -r requirements.txt

Comment Lancer les Entraînements

IMPORTANT : Toutes les commandes doivent être exécutées depuis le répertoire racine Distributed-ViT-Project/.

Étape 1 : Préparer les données (une seule fois)

python Data/prepare_data.py

Étape 2 : Lancer les entraînements

    Baseline sur un seul GPU :

    python training/single_gpu_main.py

    All-Reduce sur 4 GPUs :

    horovodrun -np 4 python training/all_reduce_main.py

    Parameter Server sur 4 GPUs (1 serveur + 3 workers) :

    horovodrun -np 4 python training/parameter_server_main.py

Critères d'Évaluation

Les résultats seront collectés pour évaluer les points suivants :

    Accélération (Speedup)

    Débit (Throughput)

    Overhead de Communication

    Scalabilité
    Critères d'Évaluation

Les résultats seront collectés pour évaluer les points suivants :

    Accélération (Speedup)

    Débit (Throughput)

    Overhead de Communication

    Scalabilité

Stratégie
	

# GPUs
	
```
Temps / Époque (s)
	

Throughput (img/s)
	

Scalabilité

Baseline
	

1
	

[X]
	

[Y]
	

1.0x

All-Reduce
	

2
	

[X]
	

[Y]
	

[Z]

All-Reduce
	

4
	

[X]
	

[Y]
	

[Z]

Param. Server
	

2 (1S+1W)
	

[X]
	

[Y]
	

[Z]

Param. Server
	

4 (1S+3W)
	

[X]
	

[Y]
	

[Z]
```