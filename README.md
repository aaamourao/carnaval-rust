# carnaval_rust

## Intro

`carnaval_rust` is a library for implementing Deep Learning models using Rust.
It is in its early days, and, by now, only predicting is possible.

The road map for the next features are:

* Load weights from files
  * The goal is to load YOLO V1 weights using `carnaval`
* Training
  * Gradient Descent

## Features

### Activation Functions

* Sigmoid
* Relu
* Leaky Relu
* Tanh
* Softmax
* *None*

### Layers

* Dense
* Conv2D (partially implemented)
* MaxPool
* Flatten (partially implemented)
* Dropout (soon)

### Model Types

* Sequential
