#!/bin/bash

# Variable name
VAR_NAME="CUDA_VISIBLE_DEVICES"

# Check if the variable exists
if [ -z "${!VAR_NAME}" ]; then
    echo "Variable does not exist. Setting it..."
    # Set the variable
    export "$VAR_NAME"="default_value"
else
    # Variable exists, use its value
    echo "Variable exists. Using value: ${!VAR_NAME}"
fi

# Print the value of the variable
echo "Final value of $VAR_NAME: ${!VAR_NAME}"


