# disaster-tweets-brane

## Introduction

This project features an implementation of an NLP pipeline for [the disaster tweets Kaggle competition](https://www.kaggle.com/competitions/nlp-getting-started/overview/description) using the [Brane](https://github.com/epi-project/brane) framework. The implementation is divided into the following Brane packages which can be imported individually and used in other workflows: `compute`, `visualization`, and `utils`.

- `compute` exposes utilities for preprocessing data, training a classifier, and generating a valid submission file for the challenge.
- `visualization` provides functions to generate plots and charts based on the dataset.
- `utils` contains generic utility functions and specifically allows for downloading the dataset at runtime.

We also include a `github.yml` specification which defines an OpenAPI container that exposes a function to download arbitrary files from GitHub repositories.

## Build

Each package can be individually imported with the following command:

```bash
brane import marinoandrea/disaster-tweets-brane -c packages/<PACKAGE_NAME>
```

However, we also provide a shell script for convenience. The user can clone the repository and simply run `./build.sh all` to build all of our packages. Additionally, you also can run the following commands to build a specific package.

```bash
# build the computation package
./build.sh compute
# build the visualization package
./build.sh visualization
# build the utils package
./build.sh utils
```

Of course, you can always navigate to the package directory and run the following command to build the brane package.

```
brane build container.yml
```
