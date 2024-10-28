#!/bin/bash

target=$1

if [ ! -d "./data" ]; then
  echo "Please enter the root of the project, then re-run the script!"
  exit 1
fi

if [[ -z "$target" || "$target" == "train" ]]; then
    echo "---- Start Build Training Data ----"
    brane data build --no-links data/train/data.yml 

    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Build training data failed, please check the error message!"
        exit 1
    fi
fi

if [[ -z "$target" || "$target" == "test" ]]; then
    echo "---- Start Build Testing Data ----"
    brane data build --no-links ./data/test/data.yml
    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Build testing data failed, please check the error message!"
        exit 1
    fi
fi

echo "---- End Build ----"
exit 0
