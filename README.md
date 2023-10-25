# Barlow Twins Self-Supervised Learning

**Author:** Jessica Bender
**Date:** 25 Oct 23

## Overview

This repository contains code for implementing the Barlow Twins self-supervised learning (SSL) framework environmental modelling. The code provides the structure for training a model on a set of unlabelled data (image patches 32x32x1) and evaluating its performance.

## Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Introduction

Self-supervised learning is a powerful technique for training deep neural networks when labelled data is scarce or expensive to obtain. The Barlow Twins method focuses on maximising the similarity of representations for augmented versions of the same data point and minimising the similarity between representations of different data points. The loss function encourages the model to create features that are invariant to transformations while reducing redundancy.

## Requirements

Before using the code, make sure you have the following dependencies installed:

- Python 3.7+
- TensorFlow 2.8.2
- TensorFlow Add Ons
- NumPy
- Matplotlib

## Usage

1. Clone the repository to your local machine

2. Navigate to the project directory

3. Prepare your dataset or data. The code assumes a specific data format (e.g., 32x32 image patches) and a training/testing split. You may need to modify the data loading part to match your dataset format.

4. Run main.py

## Results

The results of training and evaluation will be saved in the project directory. This includes saved model, loss plots, and visualisations. You can adjust the save locations and naming in the code.

This code is provided under the [MIT License](LICENSE.md). You are free to use, modify, and distribute it according to the terms of the license.

---

You can modify this README to include more specific details about your project, such as the dataset used, expected results, and any additional usage instructions.
