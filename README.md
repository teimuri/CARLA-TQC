# TQC Carla
Truncated Quantile Critics is a RL algorithm provided by Stable Baseline 3 Contrib. This repository is the implementation of TQC on a vehicle in Carla simulator.
<p align="center">
     <img src="https://github.com/teimuri/CARLA-TQC/blob/main/output_sample.gif" alt="MILE driving in imagination">
     <br/> This agent can drive in "Town02" of the Carla simulator, following a randomly generated path.
     <br/> The model is prompted with a turn signal that can take three possible values: [-1, 0, 1], each corresponding to a turn direction or going straight.
     <br/> On the right, the bird-eye view is displayed, while on the left, the reconstruction from the AutoEncoder is shown.
</p>

## ⚙ Setup

Download `Carla 0.9.13` and setup it's environment.<br/>
Use `pip install -r requirments.txt` to install the requirements.<br/>
To use the model download it from <a href="https://huggingface.co/Teimuri/TQC_CARLA/tree/main"> Huggingface</a>.<br/>

## Inference

After downloading the model place it in the models folder and use `bash run.sh` to execute the code.<br/>
Due to periodic crashing of Carla, the `run.sh` will restart the process after a crash.
