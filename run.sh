#!/bin/bash

start_processes() {
    rm -fr image_logs/*
    rm -fr logs/*
    rm -fr wandb/*
    /media/carla/carla/CARLA_0.9.13/CarlaUE4.sh -carla-port=4732 -prefernvidia -windowed -vulkan -RenderOffScreen > /dev/null 2>&1 &
    carla_pid=$!
    
    # /media/carla/carla/CARLA_0.9.13/CarlaUE4.sh -carla-port=4736 -prefernvidia -windowed -vulkan -RenderOffScreen > /dev/null 2>&1 &
    # carla_pid_eval=$!

    # tmux new-session -d -s "Train" "source /home/taha/.zshrc_carla_9_13; ; sleep 10"
    python3 /media/carla/AVRL/our_ppo/train.py --models_dir "/media/carla/AVRL/our_ppo/models" &
    train_pid=$!

    # python3 /media/carla/AVRL/our_ppo/Evaluator.py &
    # evaluation_pid=$!

    echo 2222222222
    # python3 /media/carla/AVRL/our_ppo/Evaluator.py &
    # evaluation_pid=$!

}

kill_processes() {
    
    pkill -f "/media/carla/carla/CARLA_0.9.13/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4"

    pkill -f "/media/carla/AVRL/our_ppo/train.py"
    pkill -f "/media/carla/AVRL/our_ppo/Evaluator.py"


    pkill -f "wandb"

}
exting() {
    kill_processes
    pkill -f "/media/carla/AVRL/our_ppo/file_keeper.py"
    exit 0
}
# check_python() {
    
# }
# check_carla() {
#     while true; do
#         if ! ps -p $carla_pid > /dev/null; then
#             echo "Carla stopped. Restarting Carla processes..."
#             /media/carla/carla/CARLA_0.9.13/CarlaUE4.sh -carla-port=4732 -prefernvidia -windowed -vulkan -RenderOffScreen > /dev/null 2>&1 &
#             carla_pid=$!
#         fi
#         sleep 5
#     done
# }
# check_eval_carla() {
#     while true; do
#         if  then
#             echo "Eval Carla stopped. Restarting Eval Carla processes..."
#             /media/carla/carla/CARLA_0.9.13/CarlaUE4.sh -carla-port=4736 -prefernvidia -windowed -vulkan -RenderOffScreen > /dev/null 2>&1 &
#             carla_pid_eval=$!
#         fi
#         sleep 5
#     done
# }
trap 'exting' SIGINT
kill_processes
pkill -f "/media/carla/AVRL/our_ppo/train.py"

# rm -fr /media/carla/AVRL/our_ppo/evaluation_logs/*
cp /media/carla/AVRL/our_ppo/models/trained.zip /media/carla/AVRL/our_ppo/models/trained_backup.zip
rm -i /media/carla/AVRL/our_ppo/models/trained.zip
cp /media/carla/AVRL/our_ppo/models/my_buffer.pkl /media/carla/AVRL/our_ppo/models/my_buffer_backup.pkl
# Start the processes initially
start_processes


python3 /media/carla/AVRL/our_ppo/file_keeper.py &
while true; do
    if ! ps -p $train_pid > /dev/null || ! ps -p $carla_pid > /dev/null; then
        kill_processes
        sleep 5
        echo "Trained stopped. Restarting Trained processes..."
        start_processes
    fi
    sleep 5
done
# while true; do
#     sleep 5  # Check every 5 seconds (you can adjust this)

#     if ! ps -p $train_pid > /dev/null || ! ps -p $carla_pid > /dev/null;then
#         echo "Process stopped. Restarting both processes..."

#         # Kill and restart both processes
#         kill_processes
#         #rm -fr image_logs/*
#         start_processes
#     fi
# done
