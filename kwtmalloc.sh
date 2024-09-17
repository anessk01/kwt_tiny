# Author: Aness Al-Qawlaq 
# Date: 21/2/2024
# University College Dublin

#!/bin/bash

# Source files
MAIN_SOURCE="kwtmalloc.c"
HELPERS_SOURCE="helpers.c"
WEIGHTS_SOURCE="model_weights_t.c"
INPUT_SOURCE="in_ds.c"

# Output executable name
OUTPUT_FILE="kwtmalloc"

# Compiler and flags (address is for memory leak analysis)
# -fsanitize=address
COMPILER="gcc"
FLAGS="-g -Wall -fsanitize=address"

# Compile helpers.c into an object file
$COMPILER $FLAGS -c $HELPERS_SOURCE

# Compile model_weights.c into an object file
$COMPILER $FLAGS -c $WEIGHTS_SOURCE

# Compile in_ds.c into an object file
$COMPILER $FLAGS -c $INPUT_SOURCE

# Compile kwtmalloc.c and link it with helpers.o
$COMPILER $FLAGS $MAIN_SOURCE helpers.o model_weights_t.o in_ds.o -o $OUTPUT_FILE -lm

echo "Compilation complete. Executable named '$OUTPUT_FILE' created."
