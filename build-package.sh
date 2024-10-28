#!/bin/bash

target=$1

if [ ! -d "packages/compute" ]; then
  echo "Please enter the root of the project, then re-run the script!"
  exit 1
fi

if [[ -z "$target" || "$target" == "compute" ]]; then
    echo "---- Start Build The Compute Package ----"
    brane package build packages/compute/container.yml 

    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Build compute package failed, please check the error message!"
        exit 1
    fi
fi

if [[ -z "$target" || "$target" == "visualization" ]]; then
    echo "---- Start Build The Visualization Package ----"
    brane package build packages/visualization/container.yml 
    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Build visualization package failed, please check the error message!"
        exit 1
    fi
fi

echo "---- End Build ----"
exit 0
