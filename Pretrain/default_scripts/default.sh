##########
min_port=1024
max_port=65535
port=$((min_port + RANDOM % (max_port - min_port + 1)))
export OMP_NUM_THREADS=48
##########