#!/bin/bash

n_values=(10 20 50 100 200 400 800 1000 1500 2000)

for n in "${n_values[@]}"; do

    config_files=("albert_${n}_na.yaml" "albert_${n}_aug.yaml")

    for config_file in "${config_files[@]}"; do
        session_name=$(basename "$config_file" .yaml)
        screen -dmS "$session_name" bash -c "python -m scripts.main $config_file; exec bash"
    done
done
