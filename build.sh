#!/bin/bash

echo "---- Start Build ----"

target=$0

if [ ! -d "packages/compute" ]; then
  echo "Please enter the root of the project, then re-run the script!"
  exit 1
fi
echo "$targe"

if [[ $target == "compute" ]]; then
    cd packages/compute
    brane build ./container.yml 

    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Build compute package failed, please check the error message!"
    fi
elif [[ $target == "visualization" ]]; then
    cd packages/visualization
    brane build ./container.yml
    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Build visualization package failed, please check the error message!"
    fi
elif [[ $target == "utils" ]]; then
    cd packages/utils
    brane build ./container.yml 
    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Build utils package failed, please check the error message!"
    fi
else
    cd packages/compute
    brane build ./container.yml 

    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Build compute package failed, please check the error message!"
    fi


    cd ../utils
    brane build ./container.yml 
    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Build utils package failed, please check the error message!"
    fi

    cd ../visualization
    brane build ./container.yml
    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Build visualization package failed, please check the error message!"
    fi
fi

echo "---- End Build ----"