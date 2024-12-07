
from tqc import My_TQC
import torch
# import debugpy
# debugpy.listen(("localhost", 5684))  # Start a debug server
# print("Waiting")
# debugpy.wait_for_client()
# from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
# from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import os
from carenv import CarEnv,num_envs,World_Manager
import time
import argparse
import numpy as np
import shutil
# import torch
# import carla
# import threading
from multiprocessing import shared_memory
from wandb.integration.sb3 import WandbCallback
import wandb
import copy
import pickle


# Check if the FIFO already exists, if not, create it

# Write a number to the FIFO
def write_to_fifo(number,fifo_name):
    if not os.path.exists(fifo_name):
        os.mkfifo(fifo_name)
    with open(fifo_name, 'w') as fifo:
        fifo.write(str(number))  # Write the number as a string
        print(f"Written number {number} to FIFO")

def read_from_fifo(fifo_name):
    if not os.path.exists(fifo_name):
        os.mkfifo(fifo_name)
    with open(fifo_name, 'r') as fifo:
        number = fifo.read()  # Read the data from the FIFO
    print(f"number {number} read from FIFO")
    return float(number)  # Convert it back to an integer

def process1(data,shared_name):

    try:
        existing_shm = shared_memory.SharedMemory(name=shared_name)
        existing_shm.unlink()  # Unlink (delete) the previous shared memory block
        print(f"Cleaned up existing shared memory: {shared_name}")
    except FileNotFoundError:
        # Shared memory with this name doesn't exist, nothing to clean up
        pass
    # Create some large data (for example a large numpy array)
    # data = {'large_data': np.arange(10**6)}  # Large data as a NumPy array
    
    # Serialize the data to bytes
    serialized_data = pickle.dumps(data)

    # Allocate shared memory to store the serialized data
    dtype = np.uint8
    shape = (len(serialized_data),)

    # Create shared memory block for the serialized data
    shm = shared_memory.SharedMemory(name=shared_name, create=True, size=len(serialized_data))

    # Create a numpy array that maps to the shared memory block
    np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    # Copy the serialized data into the shared memory array
    np_array[:] = np.frombuffer(serialized_data, dtype=np.uint8)

    print(f"Process 1: Data stored in shared memory.")

    # Wait to give time for Process 2 to read from shared memory
    shm.close()
    # time.sleep(120)

    # Clean up shared memory when done
    print("Process 1: Shared memory unlinked.")


def process2(shared_name):
    # Name of the shared memory block created by Process 1
    print("Process 2: Reading from shared memory.")
    dtype = np.uint8

    # Attach to the existing shared memory block created by Process 1
    shm = shared_memory.SharedMemory(name=shared_name)

    # Create a numpy array that maps to the shared memory block
    np_array = np.ndarray(shm.size, dtype=dtype, buffer=shm.buf)

    # Deserialize the data from the shared memory
    serialized_data = np_array.tobytes()
    data = pickle.loads(serialized_data)
    shm.close()
    print(f"Process 2: Data read from shared memory")

    # Unlink shared memory when done (actually remove the shared memory block)
    shm.unlink()

    return data



def evaluate_model(model, env, num_episodes=5):
    """
    Evaluates the performance of a trained model.

    :param model: The trained RL model.
    :param env: The environment to test on.
    :param num_episodes: Number of episodes to test the model.
    :return: Average reward over the episodes.
    """
    rewards = []
    for episode in range(num_episodes):
        obs,info = env.reset(evaluation=True)
        done = False
        total_reward = 0
        # single_episode_rewards = []
        while not done:
            # Get the action from the model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done,truncated, info = env.step(action)
            total_reward += reward
            # single_episode_rewards.append(reward)
            
        rewards.append(total_reward)
    avg_reward = np.mean(rewards)
    print(f"Evaluation over {num_episodes} episodes: Average Reward = {avg_reward}")
    env.reset(evaluation=False)
    return avg_reward
# FIXED_DELTA_SECONDS = 0.05
    
        

# def make_env(env_id, rank, seed=0):
#     """
#     Utility function for creating multiple environments.

#     :param env_id: The ID of the environment.
#     :param rank: Rank of the environment instance (for seeding).
#     :param seed: Random seed.
#     :return: Function that initializes the environment.
#     """
#     def _init():
#         global env_locks
#         env = CarEnv(rank)  # Replace with your environment initialization

#         return env
#     return _init
run = wandb.init(
    project="sb3",
    name="tqc",
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)
if __name__ == '__main__':
    # write_to_fifo(0,fifo_name="shared_memory")
    global model
    world_manager = World_Manager()
    # Initialize environment
    world_manager.start_world()
    
    env = CarEnv(world_manager)
    # env = SubprocVecEnv([make_env("CarEnv", i) for i in range(num_envs)])

    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--models_dir', type=str, default=None)
    args = argparser.parse_args()

    # Set up log directory for TensorBoard
    logdir = f"logs/{int(time.time())}/"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Training settings
    TIMESTEPS = 20000  # Total number of steps per training iteration
    LR = 7.3e-4  # Learning rate for TQC

    # Check if a pre-trained model exists and load or create a new one
    # model = My_TQC.load(f"{args.models_dir}/trained_shared_memory.zip", env=env, verbose=1, learning_rate=LR, tensorboard_log=logdir)
    # model.replay_buffer = process2()
    # model = My_TQC.load(f"{args.models_dir}/trained_shared_memory.zip", env=env, verbose=1, learning_rate=LR, tensorboard_log=logdir)
    # model.replay_buffer = process2()
    # fifo_signal = read_from_fifo(fifo_name="shared_memory")
    fifo_signal = 0
    models_dir = args.models_dir
    if fifo_signal == 1:
            print("Loading from shared memory")
            model = My_TQC.load(f"{args.models_dir}/trained_shared_memory.zip", env=env, verbose=1, learning_rate=LR, tensorboard_log=logdir)
            model.replay_buffer = process2(shared_name="buffer_shared_memory")
            # print(found)
    else:
        print("Creating new model")
        if args.models_dir is None or (not os.path.isfile(f"{args.models_dir}/trained.zip")):
            
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)

            # Initialize TQC model
            model = My_TQC(
                'MlpPolicy',
                env=env,
                buffer_size=200_000,
                batch_size=256,
                train_freq=200,
                gradient_steps=256,
                learning_starts=500,
                use_sde=True,
                sde_sample_freq=16,
                policy_kwargs=dict(
                    log_std_init=-3,
                    net_arch=[256, 256],
                    n_critics=2,
                    use_expln=True
                ),
                verbose=1,
                learning_rate=LR,
                gamma=0.99,  # Adjust gamma if needed
                tensorboard_log=logdir,
                # device='cuda',
                # ent_coef=2
                
            )
            model.replay_buffer.avg_rewards = [0]
            model.replay_buffer.num_saved = 0
        else:
            # raise ValueError(999999999999)
            models_dir = args.models_dir
            
            model = My_TQC.load(f"{models_dir}/trained.zip", env=env, verbose=1, learning_rate=LR, tensorboard_log=logdir)
            
            # model.log_ent_coef = torch.log(torch.ones(1, device=model.device) * 0.1).requires_grad_(True)
            # model.ent_coef_optimizer = torch.optim.Adam([model.log_ent_coef], lr=model.lr_schedule(1))


            replay_buffer = pickle.load(open(f"{models_dir}/replay_buffer.pkl", "rb"))
            model.replay_buffer = replay_buffer
            # model.replay_buffer.avg_rewards = replay_buffer.avg_rewards
            # model.replay_buffer.num_saved = replay_buffer.num_saved
            del replay_buffer
            # model.load_replay_buffer(f"{models_dir}/replay_buffer.pkl")
        # model.ent_coef = 'auto_0.1'
        # model._setup_model()
        # coef = 1
        # model.ent_coef = coef
        # model.ent_coef_tensor = torch.tensor(coef)
    # Connect to the environment and reset it
    print('Connecting to environment...')
    print('Environment has been reset as part of launch')

    # Training loop
    iters = 0

    # model_copy = copy.deepcopy(model)
    for iters in range(0,20):  # Number of training iterations
        print(f'Iteration {iters} is commencing...')
        # model.replay_buffer.iters = 33
        # Train the model for a specified number of timesteps

        # model.save(f"{models_dir}/trained_shared_evaluation.zip")
        last_replay_buffer = copy.deepcopy(model.replay_buffer)
        # process1(last_replay_buffer,shared_name="evaluation")
        # write_to_fifo(1,fifo_name="evaluation")
        # read_from_fifo(fifo_name="evaluation")
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,callback=WandbCallback(verbose=2))
        model.save(f"{models_dir}/trained.zip")
        model.save_replay_buffer(f"{models_dir}/replay_buffer.pkl")
        # raise ValueError(iters)
        # avg_reward = read_from_fifo(fifo_name="evaluation")
        # model.replay_buffer.avg_rewards.append(avg_reward)
        
        # last_model.replay_buffer.avg_rewards.append(avg_reward)

        # model.save(f"models/trained_shared_memory.zip")
        # process1(model.replay_buffer,shared_name="buffer_shared_memory")
        # write_to_fifo(1,fifo_name="shared_memory")
        # read_from_fifo(fifo_name="shared_memory")
        
        # if (np.max(last_replay_buffer.avg_rewards) < avg_reward):
        # last_replay_buffer.num_saved += 1
        # shutil.copy(f"{models_dir}/trained_shared_evaluation.zip", f"{models_dir}/trained.zip")
        # pickle.dump(last_replay_buffer, open(f"{models_dir}/replay_buffer.pkl", "wb"))
        # run.log({'Evaluation Reward': avg_reward,'iter':len(model.replay_buffer.avg_rewards)-1})
        
        print(f'Iteration {iters} has been trained')
        # model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,callback=WandbCallback(verbose=2))
        # except Exception as e:
        #     from tqc import My_TQC
        #     world_manager.start_world()
        #     env = CarEnv(0)
        #     model = My_TQC.custom_load(saved_data = data_copy, env=env, verbose=1, learning_rate=LR, tensorboard_log=logdir)
        #     model.replay_buffer = replay_buffer_copy
        #     print('Error occurred during training:', e)

        # Save the trained model after each iteration
        
        # model.save_replay_buffer(f"{models_dir}/my_buffer.pkl")
        
        
        
