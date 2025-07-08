Distributed Vision Transformer (ViT) Project

This repository contains the code and resources for implementing and training a Vision Transformer (ViT) model in a distributed manner. The project explores various distributed training strategies to efficiently scale the training of large-scale vision models on multi-GPU or multi-node systems.
Table of Contents

    Introduction

    Features

    System Architecture

    Getting Started

        Prerequisites

        Installation

    Usage

        Data Preparation

        Running the Training

    Distributed Training Strategies

    Results

    Contributing

    License

    Acknowledgments

Introduction

The Vision Transformer (ViT) has emerged as a state-of-the-art model for computer vision tasks. However, training these large models requires significant computational resources. This project aims to address this challenge by implementing a distributed training pipeline for the ViT model. By leveraging frameworks like PyTorch's DistributedDataParallel (DDP), we can significantly reduce training time and enable the training of even larger models.

The primary goals of this project are:

    To provide a clear and well-documented implementation of a Vision Transformer.

    To demonstrate how to set up and run distributed training on a target dataset (e.g., ImageNet, CIFAR-100).

    To compare the performance and scalability of different distributed strategies.

Features

    Vision Transformer Implementation: A clean implementation of the ViT architecture from scratch.

    Distributed Training: Support for multi-GPU and multi-node training using torch.distributed.

    Multiple Strategies: Code to demonstrate different parallelization techniques (e.g., Data Parallelism).

    Scalability: Designed to scale efficiently with an increasing number of processing units.

    Easy Configuration: Training parameters, model architecture, and distributed settings can be easily configured.

System Architecture

This project uses a standard distributed training setup. The architecture consists of multiple processes (workers), each controlling a single GPU. The model is replicated across all GPUs, and each process receives a unique shard of the training data. Gradients are synchronized across all processes after the backward pass using the all-reduce algorithm.

(Optional: You can add a simple diagram here to illustrate the architecture if you'd like.)
Getting Started

Follow these instructions to set up the project on your local machine or cluster.
Prerequisites

    Python 3.8+

    PyTorch 1.9+

    CUDA 11.0+

    [Add any other major dependencies, e.g., torchvision, numpy, etc.]

Installation

    Clone the repository:

    git clone https://github.com/cchahid/Distributed-ViT-Project.git
    cd Distributed-ViT-Project


    Create a virtual environment (recommended):

    python -m venv venv
    source venv/bin/activate


    Install the required packages:
    (Please create a requirements.txt file for a better user experience)

    pip install -r requirements.txt


Usage
Data Preparation

    Download the [Your Dataset Name, e.g., CIFAR-100] dataset.

    Place it in the data/ directory or specify the path in the configuration file.

    [Add any specific preprocessing steps if necessary]

Running the Training

The training script is launched using torchrun (or torch.distributed.launch). This utility handles setting up the distributed environment.

To run the training on a single machine with N GPUs (e.g., 4 GPUs):

torchrun --nproc_per_node=4 train.py --batch_size 32 --epochs 100 --lr 1e-4


Key Arguments:

    --nproc_per_node: The number of GPUs to use on the current machine.

    --batch_size: The batch size per GPU.

    [Add other important command-line arguments you have, like learning rate, model size, etc.]

Distributed Training Strategies

This project primarily implements Data Parallelism using torch.nn.parallel.DistributedDataParallel (DDP).

    How it works: The model is copied to every GPU. The dataset is split, and each GPU processes its own mini-batch. The gradients are then averaged across all GPUs before the optimizer updates the weights, ensuring all models remain synchronized.

(If you implement other strategies like model parallelism or pipeline parallelism, describe them here.)
Results

(This is a crucial section. After you run your experiments, fill this in with your findings.)

    Training Performance: Include a table or graph showing training time vs. the number of GPUs.

    Scalability: Plot the training throughput (e.g., images/second) as you increase the number of workers.

    Model Accuracy: Report the final accuracy of the trained model on the test set.

| # GPUs | Training Time (hours) | Throughput (images/sec) | Accuracy (%) |
| 1 | [X] | [Y] | [Z] |
| 2 | [X] | [Y] | [Z] |
| 4 | [X] | [Y] | [Z] |
Contributing

Contributions are welcome! If you have suggestions for improving this project, please feel free to open an issue or submit a pull request.

    Fork the Project

    Create your Feature Branch (git checkout -b feature/AmazingFeature)

    Commit your Changes (git commit -m 'Add some AmazingFeature')

    Push to the Branch (git push origin feature/AmazingFeature)

    Open a Pull Request

License

This project is licensed under the [Your License, e.g., MIT License]. See the LICENSE file for more details.
Acknowledgments

    This implementation is based on the original paper: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

    The PyTorch documentation on distributed training was an invaluable resource.

    [Any other libraries, articles, or people you want to thank]