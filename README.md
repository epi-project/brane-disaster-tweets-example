# disaster-tweets-brane

## Introduction
This project contains three sub-projects: `compute`, `visualization`, and `utils`.
The `compute` is a data process brane package for [the disaster tweets Kaggle competition](https://www.kaggle.com/competitions/nlp-getting-started/overview/description) that contains preprocess data, training model, and prediction. The `visualization` is a brane package to generate all the plots based on the dataset. The `utils` is a utility brane package that provides general features like download files.

***Note:*** This project is based on the [brane](https://github.com/epi-project/brane) and uses the ***Git Submodules*** to do vision control.

## Build Package

We provide a shell script called build.sh to help users build our packages. 
You can run ``` ./build.sh all ```  to build all of our packages. Aside from that, you also can run the following commands to build one of our packages.

```bash
#build the computation package
./build.sh compute
#build the visualization package
./build.sh visualization
#build the utils package
./build.sh utils
```

After build package, you can deploy our package through the brane. 