# Natural Language Processing with Disaster Tweets

[![DOI](https://zenodo.org/badge/491528415.svg)](https://zenodo.org/badge/latestdoi/491528415)
![Compute Package](https://github.com/marinoandrea/disaster-tweets-brane/actions/workflows/compute-package.yaml/badge.svg)
![Visualization Package](https://github.com/marinoandrea/disaster-tweets-brane/actions/workflows/visualization-package.yaml/badge.svg)

## Introduction

This project features an implementation of an NLP pipeline for [the disaster tweets Kaggle competition](https://www.kaggle.com/competitions/nlp-getting-started/overview/description) using the [Brane](https://github.com/epi-project/brane) framework. The implementation is divided into the following Brane packages which can be imported individually and used in other workflows: `compute` and `visualization`.

- `compute` exposes utilities for preprocessing data, training a classifier, and generating a valid submission file for the challenge.
- `visualization` provides functions to generate plots and charts based on the dataset.

We also include a `github.yml` specification which defines an OpenAPI container that exposes a function to download arbitrary files from GitHub repositories.

## Build

Each package can be individually imported with the following command:

```bash
brane package import -c epi-project/brane-disaster-tweets-example packages/<PACKAGE_NAME>/container.yml
```

However, we also provide a shell script for convenience. The user can clone the repository and simply run `./build-package.sh` to build all of our packages. Additionally, you also can run the following commands to build a specific package.

```bash
# build the computation package
./build-package.sh compute
# build the visualization package
./build-package.sh visualization
```

Of course, you can always navigate to the package directory and run the following command to build the brane package.

```bash
brane package build container.yml
```

## Data

Besides packages, we also need to build the datasets used by the workflow. This can be done using the included `./build-data.sh` script to build the training and testing dataset.

```bash
# For the training dataset
brane data build ./data/train/data.yml
# For the testing dataset
brane data build ./data/test/data.yml
```

## Run

Our pipeline implementation can be executed locally by simply running the following command in the root folder of the project:

```bash
brane workflow run pipeline.bs
```
The following picture shows an example that our package uses the pipeline.bs to run the whole pipeline in the Kubernetes cluster.
![Example Runs On Kubernetes cluster](WX20220603-195559.png)  


## Attribution
This repository is the up-to-date version of the work of Andrea Marino and Jingye Wang, with the aim to implement exactly the same as they have done for a newer version of the framework. Their original repository can be found [here](https://github.com/marinoandrea/disaster-tweets-brane).
