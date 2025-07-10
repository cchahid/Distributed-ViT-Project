Comparaison de Stratégies d'Optimisation Distribuée pour Vision Transformer

Ce projet a pour objectif d'évaluer et de comparer l'efficacité de deux stratégies de formation distribuée — All-Reduce et Parameter Server — par rapport à une baseline sur un seul GPU, pour l'entraînement d'un modèle Vision Transformer (ViT).
Table des Matières

    Objectif du Projet

    Architectures Étudiées

        Baseline (Single GPU)

        All-Reduce (Décentralisé)

        Parameter Server (Centralisé)

    Structure du Projet

    Mise en Route

    Comment Lancer les Entraînements

    Critères d'Évaluation

    Contribuer

    Licence

Objectif du Projet

L'objectif principal est de mesurer et d'analyser les performances de différentes architectures de communication pour la synchronisation des gradients lors de l'entraînement d'un grand modèle de vision. Nous cherchons à répondre aux questions suivantes :

    Quelle stratégie offre la meilleure accélération (speedup) lorsque le nombre de GPU augmente ?

    Quel est le surcoût (overhead) de communication de chaque méthode ?

    Comment la scalabilité est-elle affectée par l'architecture de communication ?

Architectures Étudiées
Baseline (Single GPU)

Il s'agit de l'entraînement standard sur une seule carte graphique, sans aucune parallélisation. C'est la référence par rapport à laquelle les performances des autres stratégies seront mesurées.

    Implémentation : training/single_gpu_main.py.

All-Reduce (Décentralisé)

Dans cette architecture, chaque processus (GPU) communique ses gradients à tous les autres processus. Il n'y a pas de serveur central. Les gradients sont moyennés de manière décentralisée, souvent via un algorithme en anneau (ring-allreduce).

    Implémentation : training/all_reduce_main.py (basé sur hvd.DistributedOptimizer).

    Avantages : Évite le goulot d'étranglement d'un serveur central.

    Inconvénients : Le coût de communication peut augmenter avec le nombre de nœuds.

Parameter Server (Centralisé)

Cette architecture utilise un processus dédié (le Parameter Server, rang 0) qui est seul responsable de la mise à jour des poids du modèle. Les autres processus (les workers) calculent les gradients sur leurs lots de données et les envoient au serveur.

    Implémentation : training/parameter_server_main.py.

    Avantages : Simplifie la synchronisation. Peut être efficace pour des mises à jour asynchrones.

    Inconvénients : Le serveur peut devenir un goulot d'étranglement réseau.

Structure du Projet

Distributed-ViT-Project/
│
├── Data/
│   ├── prepare_data.py
│   └── tiny-imagenet-200/     # (Données après exécution du script)
│
├── training/
│   ├── single_gpu_main.py
│   ├── all_reduce_main.py
│   └── parameter_server_main.py
│
├── requirements.txt
├── README.md
└── LICENSE

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

    Note : L'installation de Horovod peut nécessiter des étapes supplémentaires. Consultez la documentation officielle de Horovod.

Comment Lancer les Entraînements

IMPORTANT : Toutes les commandes doivent être exécutées depuis le répertoire racine Distributed-ViT-Project/.

Étape 1 : Préparer les données
(Cette commande ne doit être exécutée qu'une seule fois)

python Data/prepare_data.py

Étape 2 : Lancer les entraînements

    Lancer la baseline sur un seul GPU :

    python training/single_gpu_main.py

    Lancer l'entraînement All-Reduce sur 4 GPUs :

    horovodrun -np 4 python training/all_reduce_main.py

    Lancer l'entraînement Parameter Server sur 4 GPUs (1 serveur + 3 workers) :

    horovodrun -np 4 python training/parameter_server_main.py

Critères d'Évaluation

Les résultats seront collectés pour évaluer les points suivants :

    Accélération (Speedup) : Comparaison du temps d'entraînement total par rapport à la baseline.

    Débit (Throughput) : Mesure du nombre d'images traitées par seconde.

    Overhead de Communication : Analyse du temps passé dans les opérations de communication.

    Scalabilité : Efficacité de la parallélisation (Speedup / Nombre de GPUs).

Stratégie
	

# GPUs
	

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