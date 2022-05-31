#!/bin/bash


target=$1

if [ ! -d "packages/compute" ]; then
  echo "Please enter the root of the project, then re-run the script!"
  exit 1
fi

if [[ $target == "compute" ]]; then
    echo "---- Start Build The Compute Package ----"
    cd packages/compute
    brane build ./container.yml 

    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Build compute package failed, please check the error message!"
        exit 1
    fi
elif [[ $target == "visualization" ]]; then
    echo "---- Start Build The Visualization Package ----"
    cd packages/visualization
    brane build ./container.yml
    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Build visualization package failed, please check the error message!"
        exit 1
    fi
elif [[ $target == "utils" ]]; then
    echo "---- Start Build The Utils Package ----"
    cd packages/utils
    brane build ./container.yml 
    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Build utils package failed, please check the error message!"
        exit 1
    fi
else
    echo "---- Start Build All The Package ----"
    cd packages/compute
    brane build ./container.yml 

    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Build compute package failed, please check the error message!"
        exit 1
    fi


    cd ../utils
    brane build ./container.yml 
    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Build utils package failed, please check the error message!"
        exit 1
    fi

    cd ../visualization
    brane build ./container.yml
    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Build visualization package failed, please check the error message!"
        exit 1
    fi
fi

echo "---- End Build ----"
exit 0