# Smart Traffic Signal Control using YOLOv8 and Deep Reinforcement Learning
## Overview

Traffic congestion is a major challenge in modern cities. Traditional traffic signals operate using fixed timing schedules that do not adapt to real-time traffic conditions, leading to inefficient traffic flow and increased waiting times.

This project presents an AI-driven adaptive traffic signal control system that integrates computer vision and reinforcement learning. The system uses YOLOv8 to detect vehicles from traffic images or video and estimate lane-wise vehicle counts.

These counts are used as the state representation for a Deep Q-Network (DQN) agent, which learns optimal traffic signal timings to reduce congestion.

A Streamlit-based visualization module simulates the traffic intersection and displays the decisions made by the reinforcement learning agent.

## System Architecture

The system pipeline consists of the following stages:

```
Traffic Image / Video
        |
        v
Vehicle Detection using YOLOv8
        |
        v
Lane-wise Vehicle Counting
        |
        v
Traffic State Representation
        |
        v
Deep Q-Network (DQN)
        |
        v
Optimal Signal Decision
        |
        v
Traffic Simulation Visualization
```

## Key Features

- Vehicle detection using YOLOv8

- Lane-wise vehicle counting

- Reinforcement learning based traffic signal optimization

- Queue-length based reward mechanism

- Real-time traffic intersection simulation using Streamlit

- Visualization of reinforcement learning decisions

## Technologies Used

- Python

- PyTorch

- YOLOv8 (Ultralytics)

- OpenCV

- Gymnasium

- Streamlit

- Pandas

- NumPy

- Shapely

## Project Structure
```
SmartTraffic-RL
│
├── yolo.py
│   Vehicle detection and lane-wise counting using YOLOv8
│
├── traffic_dqn.py
│   Reinforcement learning environment and DQN training
│
├── visualize.py
│   Streamlit dashboard for traffic simulation
│
├── lane_vehicles.csv
│   Extracted vehicle counts per lane
│
├── rl_output.txt
│   Reinforcement learning output and signal decisions
│
└── README.md
```
## Installation

Clone the repository

git clone [https://github.com/moneeshas/Vision-Based-Adaptive-Traffic-Signal-Control-with-Deep-Reinforcement-Learning.git](https://github.com/moneeshas/Vision-Based-Adaptive-Traffic-Signal-Control-with-Deep-Reinforcement-Learning)

cd Vision-Based-Adaptive-Traffic-Signal-Control-with-Deep-Reinforcement-Learning

Install the required dependencies

pip install ultralytics opencv-python pandas numpy torch gymnasium streamlit shapely

## How to Run the Project
Step 1: Vehicle Detection

Run the YOLO detection script:

python yolo.py

This script:

- Detects vehicles in the traffic image

- Counts vehicles per lane

- Stores results in lane_vehicles.csv

Step 2: Train the Reinforcement Learning Model

Run the reinforcement learning script:

python traffic_dqn.py

The DQN agent will:

- Learn optimal signal timings

- Reduce traffic congestion based on queue lengths

- Save decisions in rl_output.txt

Step 3: Visualize the Traffic Simulation

Run the Streamlit application:

streamlit run visualize.py

- This launches a browser-based dashboard that visualizes the traffic signal simulation and RL decisions.

## Reinforcement Learning Model
State

The state represents the queue length of vehicles in each lane.

Example:

[33, 27, 24, 23]

Actions

Each action represents selecting a lane to receive the green signal.

Reward Function

The reward function is designed to minimize traffic congestion.

reward = throughput_reward 
       + priority_bonus 
       + empty_lane_penalty 
       + imbalance_penalty 
       + wait_penalty

Where:

- throughput_reward – total vehicles that passed the signal

- wait_penalty – penalty if waiting time exceeds threshold

- imbalance_penalty – queue imbalance across lanes

- priority_bonus – reward for prioritizing highest density lane

- empty_lane_penalty – penalty if green signal is given to empty lane

This formulation ensures the agent maximizes throughput while minimizing waiting and congestion simultaneously.

## Example Output

Example reinforcement learning output:

Step 495 | Lane 1 | Max Queue Length: 33 | Green Time: 21s | Queues [33,27,24,23]

## Future Improvements

- Multi-agent reinforcement learning

- Deployment on edge devices

- Emergency Vehicle Detection

- Pedestrian and Cyclist Detection
