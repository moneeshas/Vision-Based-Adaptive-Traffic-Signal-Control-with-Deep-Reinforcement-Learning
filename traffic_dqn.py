import os   
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
import random
from collections import deque, namedtuple
import pandas as pd
import numpy as np

# ---------------------------
# Replay Buffer & DQN
# ---------------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=[128,128]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_dim, action_dim, device='cpu', lr=1e-3, gamma=0.99,
                 buffer_size=50000, batch_size=64, target_update=1000):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.net = DQNNet(obs_dim, action_dim).to(device)
        self.target = DQNNet(obs_dim, action_dim).to(device)
        self.target.load_state_dict(self.net.state_dict())
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.replay = ReplayBuffer(capacity=buffer_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 100000

        self.steps_done = 0
        self.target_update = target_update

    def select_action(self, state, eval_mode=False):
        eps = self.eps_end if eval_mode else(self.eps_end + (self.eps_start - self.eps_end)*\
            np.exp(-1. * self.steps_done / self.eps_decay))
        self.steps_done += 1
        if random.random() > eps or eval_mode:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q = self.net(s)
                action = int(q.argmax().item())
        else:
            action = random.randrange(self.action_dim)
        return action

    def push_transition(self, *args):
        self.replay.push(*args)

    def update(self):
        if len(self.replay) < self.batch_size:
            return 0.0
        batch = self.replay.sample(self.batch_size)
        state = torch.tensor(np.vstack(batch.state), dtype=torch.float32, device=self.device)
        action = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(np.vstack(batch.next_state), dtype=torch.float32, device=self.device)
        done = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.net(state).gather(1, action)
        with torch.no_grad():
            next_q = self.target(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + (1 - done) * self.gamma * next_q

        loss = nn.functional.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target.load_state_dict(self.net.state_dict())

        return loss.item()

# ---------------------------
# Gymnasium Traffic Environment
# ---------------------------
class TrafficSignalEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, num_lanes=4, max_queue=40, min_green=5, yellow_time=2, max_episode_steps=50,use_penalties=True):
        super().__init__()
        self.num_lanes = num_lanes
        self.max_queue = max_queue
        self.min_green = min_green
        self.yellow_time = yellow_time
        self.max_episode_steps = max_episode_steps
        self.use_penalties = use_penalties

        self.observation_space = spaces.Box(low=0, high=max_queue, shape=(num_lanes+2,), dtype=np.float32)
        self.action_space = spaces.Discrete(1 + num_lanes)

        self.queues = np.zeros(num_lanes, dtype=np.int32)
        self.current_green = 0
        self.time_in_green = 0
        self.phase = 'green'
        self.yellow_counter = 0
        self.step_count = 0
        self.empty_lane_counter = 0 
        self.departure_rate = 2

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.queues[:] = 0
        self.current_green = 0
        self.time_in_green = 0
        self.phase = 'green'
        self.yellow_counter = 0
        self.empty_lane_counter = 0 
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.queues.astype(np.float32),
                               np.array([float(self.current_green), float(self.time_in_green)], dtype=np.float32)])


    def step(self, action, external_counts=None):
        self.step_count += 1
        switching_initiated = False
        target_lane = None
        # MAX_GREEN_CYCLE SHOULD BE ONE CYCLE PER LANE

# If the current green lane has exceeded its allowed green cycles
        # if self.time_in_green >= MAX_GREEN_CYCLE:
        #     self.phase = 'yellow'
        #     self.yellow_counter = self.yellow_time
        #     self.time_in_green = 0

        #     # Choose next lane with highest queue (not the same one)
        #     next_lane = int(np.argmax(self.queues)!=self.current_green)
        #     if next_lane == self.current_green:
        #         # if the current one is still max, pick the next one cyclically
        #         next_lane = (self.current_green + 1) % self.num_lanes

        #     self.current_green = next_lane
        #     switching_initiated = True


    # # Force agent to pick the lane with the highest number of vehicles if current green lane has 0
    #     if np.all(self.queues == 0):
    #         reward=1.0
    #     # All lanes empty: stay idle safely
    #         target_lane = self.current_green
    #     else:
    #         #if current lane's vehicular count is 0
    #         if self.queues[self.current_green] == 0:
    #         # Force switch to lane with max queue
    #             target_lane = int(np.argmax(self.queues))
    #             #if target lane is not green, switch the current lane to yellow and set the target lane to green
    #             if target_lane != self.current_green:
    #                 self.phase = 'yellow'
    #                 self.yellow_counter = self.yellow_time
    #                 switching_initiated = True
    #                 target_lane= self.current_green
    #         else:
    #             # Force switch to max-queue lane if current lane is empty or has very low count
    #             if self.queues[self.current_green] < 3 and np.max(self.queues) > 5:
    #                 target_lane = int(np.argmax(self.queues))
    #                 if target_lane != self.current_green:
    #                     self.phase = 'yellow'
    #                     self.yellow_counter = self.yellow_time
    #                     switching_initiated = True
    #                     self.current_green = target_lane

            # Normal case: obey agent's selected action
        if action != 0:
            target_lane = action - 1
            if target_lane >= self.num_lanes:
                target_lane = self.current_green
            if target_lane != self.current_green and self.phase == 'green' and self.time_in_green >= self.min_green:
                self.phase = 'yellow'
                self.yellow_counter = self.yellow_time
                switching_initiated = True
            else:
                target_lane = self.current_green
        else:
            target_lane = self.current_green


    #  Departures
        departed = np.zeros(self.num_lanes, dtype=np.int32)
        if self.phase == 'green':
    # Dynamically adjust green time based on current queue
            dynamic_green_time = self.min_green + int(10 * (self.queues[self.current_green] / (self.max_queue + 1)))
            dynamic_green_time = np.clip(dynamic_green_time, 5, 25)

        
            depart = min(self.queues[self.current_green], self.departure_rate)
            departed[self.current_green] = depart
            self.queues[self.current_green] -= depart
            self.time_in_green += 1

        # Once the green time exceeds the dynamic limit, prepare to switch
            if self.time_in_green >= dynamic_green_time:
                self.phase = 'yellow'
                self.yellow_counter = self.yellow_time

        elif self.phase == 'yellow':
            depart = max(0, min(self.queues[self.current_green], int(self.departure_rate * 0.5)))
            departed[self.current_green] = depart
            self.queues[self.current_green] -= depart
            self.yellow_counter -= 1
            if self.yellow_counter <= 0:
                if target_lane is not None and target_lane != self.current_green:
                    self.current_green = target_lane
                self.phase = 'green'
                self.time_in_green = 0
           # --- Handle arrivals and queue update ---
        if external_counts is not None:
            # YOLO mode: directly set queue counts from external vehicle data
            self.queues = np.array(external_counts[:self.num_lanes], dtype=np.int32)
        else:
            # Normal simulation mode: random arrivals + queue dynamics
            arrivals = np.random.poisson(1.5, size=self.num_lanes)
            self.queues = np.minimum(self.max_queue, self.queues + arrivals - departed)

        # Track how long current green lane has been empty
        if self.queues[self.current_green] == 0:
            self.empty_lane_counter += 1
        else:
            self.empty_lane_counter = 0

        # if self.step_count % self.min_green == 0:
        #     self.current_green = (self.current_green + 1) % self.num_lanes    
# 🚦 Smart dynamic lane switching based on YOLO/queue density
        if external_counts is not None:
            # Use YOLO-provided counts directly
            self.queues = np.array(external_counts[:self.num_lanes], dtype=np.int32)

        # Choose the lane with the maximum number of vehicles
        max_lane = int(np.argmax(self.queues))
        self.current_green = max_lane
        # Dynamic green duration proportional to vehicle density
        max_queue = np.max(self.queues)
        self.dynamic_green_time = max(5, min(25, int(5 + 0.5 * max_queue)))


 
        # Apply penalty if lane is empty for more than 3 steps
        empty_lane_duration_penalty = 0.0
        if self.empty_lane_counter > 3:  # > 3 steps
            empty_lane_duration_penalty = -2.0  # You can adjust this value

            
    # --- Reward and logging ---
        # Compute key metrics
        total_wait = float(self.queues.sum())
        throughput = float(departed.sum())
        max_lane = int(np.argmax(self.queues))
        max_queue = float(self.queues[max_lane])

        # --- New Reward Function ---
        # Encourage switching to the lane with the highest queue
        priority_bonus = 4.0 if self.current_green == max_lane else -2.0
        empty_lane_penalty = -5.0 if self.queues[self.current_green] == 0 else 0.0
        imbalance_penalty = -0.1 * np.std(self.queues) if self.use_penalties else 0.0
        wait_penalty = -0.05 * total_wait
        throughput_reward = 0.5 * throughput

        reward = throughput_reward + priority_bonus + empty_lane_penalty + imbalance_penalty + wait_penalty
        reward = np.clip(reward, -20, 20)




        terminated = self.step_count >= self.max_episode_steps
        truncated = False
        info = {'queues': self.queues.copy(), 'throughput': throughput, 'switching_initiated': switching_initiated}

        return self._get_obs(), reward, terminated, truncated, info



    def render(self, mode='human'):
        pass


from gymnasium.vector import AsyncVectorEnv

def make_env_fn(num_lanes=4, use_penalties=True):
    def _init():
        return TrafficSignalEnv(num_lanes=num_lanes, use_penalties=use_penalties)
    return _init

# ---------------------------
# Training Loop (Gymnasium ready)
# ---------------------------
def train(num_episodes=1000, save_interval=1000):
    NUM_ENVS = 8
    NUM_LANES=4
    env_fns = [make_env_fn(num_lanes=NUM_LANES) for _ in range(NUM_ENVS)]
    env = AsyncVectorEnv(env_fns)

    obs_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n


    # env = TrafficSignalEnv(num_lanes=NUM_LANES, min_green=5, yellow_time=2)
    # obs_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda is available" if device.type == "cuda" else "On CPU")

    agent = DQNAgent(obs_dim, action_dim, device=device, lr=1e-3)

    checkpoint_path = "dqn_traffic_checkpoint.pth"

    # Try to resume training if checkpoint exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.net.load_state_dict(checkpoint["model_state_dict"])
        agent.target.load_state_dict(checkpoint["target_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.steps_done = checkpoint.get("steps_done", 0)
        start_episode = checkpoint.get("episode", 0)
        print(f"Resumed training from episode {start_episode}")
    else:
        print("Starting fresh training")
        start_episode = 0

    all_rewards = []

    # ================================
    # Training loop
    # ================================
    for ep in range(start_episode, num_episodes):
        states, _ = env.reset()
        episode_rewards = np.zeros(NUM_ENVS)

        max_steps = 500
        for step in range(max_steps):
            actions = [agent.select_action(s) for s in states]
            next_states, rewards, terminated, truncated, infos = env.step(actions)
            dones = np.logical_or(terminated, truncated)

            for i in range(NUM_ENVS):
                agent.push_transition(states[i], actions[i], rewards[i], next_states[i], float(dones[i]))
                episode_rewards[i] += rewards[i]

            agent.update()
            states = next_states

            if np.all(dones):
                break

        avg_reward = np.mean(episode_rewards)
        all_rewards.append(avg_reward)


        # Logging
        if (ep + 1) % 500== 0:
            avg_reward = np.mean(all_rewards[-500:])
            print(f"Episode {ep + 1}/{num_episodes} | Avg reward (last50): {avg_reward:.2f}")

        # Save checkpoint every few episodes
        if (ep + 1) % save_interval == 0 or (ep + 1) == num_episodes:
            checkpoint = {
                "episode": ep + 1,
                "model_state_dict": agent.net.state_dict(),
                "target_state_dict": agent.target.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "steps_done": agent.steps_done,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at episode {ep + 1}")

    print("Training completed!")
    print(f"Final model saved at: {checkpoint_path}")

    return agent, env


def evaluate(agent, env, num_episodes=5, render=True, use_penalties=False):
    print("\n--- Evaluation Phase ---")
    all_rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        terminated = False
        while not terminated:
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            total_reward += reward
            if render:
                env.render()
        all_rewards.append(total_reward)
        print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}")
    avg_reward = np.mean(all_rewards)
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}")
    return all_rewards



def evaluate_with_yolo(agent, env, yolo_csv='lane_vehicles.csv', render=True):
    """
    Evaluate the trained RL agent on vehicle counts from a YOLO CSV.
    Logs step info to console and rl_output.txt, returns average reward.
    """
    # import pandas as pd
    # from collections import deque
    # import numpy as np
    # import os

    if not os.path.exists(yolo_csv):
        raise FileNotFoundError(f"YOLO CSV file not found at: {yolo_csv}")

    df = pd.read_csv(yolo_csv)
    lane_cols = [col for col in df.columns if 'lane' in col.lower()]
    recent_lanes = deque(maxlen=4)
    total_reward = 0.0
    env.reset()

    for frame_idx, row in df.iterrows():
        lane_counts = [int(row[col]) for col in lane_cols]
        
        # pad/truncate to match env lanes
        if len(lane_counts) < env.num_lanes:
            lane_counts += [0] * (env.num_lanes - len(lane_counts))
        elif len(lane_counts) > env.num_lanes:
            lane_counts = lane_counts[:env.num_lanes]
        
        queues = np.array(lane_counts, dtype=int)

        # Pick lane with max queue among candidates
        candidate_lanes = [i for i in range(env.num_lanes) if i not in recent_lanes]
        if not candidate_lanes:
            candidate_lanes = list(range(env.num_lanes))
        agent_action = max(candidate_lanes, key=lambda i: queues[i])
        max_queue_length = queues[agent_action]

        # Step env
        next_obs, reward, terminated, truncated, info = env.step(agent_action + 1, external_counts=queues)
        total_reward += reward

        # Update recent lanes
        recent_lanes.append(agent_action)

        green_time = env.time_in_green
        print(f"Step {frame_idx+1:4d} | Lane {agent_action+1} | Max_Queue_length: {max_queue_length} | Green Time: {green_time}s | Queues {lane_counts} | Reward {total_reward:.2f}")

        # Save to file
        with open("rl_output.txt", "a") as f:
            f.write(f"Step {frame_idx+1} | Lane {agent_action+1} | Max_Queue_length: {max_queue_length} | Green Time: {green_time}s | Queues {lane_counts} | Reward {total_reward:.2f}\n")

        if terminated:
            env.reset()


        if render:
            env.render()

    avg_reward = total_reward / len(df)
    print(f"\nEvaluation completed. Average reward: {avg_reward:.2f}")
    return avg_reward




# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# def animate_traffic(env, agent=None, steps=200, interval=400):
#     """
#     Animated visualization of the traffic simulation.
#     If agent is provided, actions are chosen by the trained agent.
#     Otherwise, random actions are used.
#     """
#     num_lanes = env.num_lanes
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_xlim(-3, 3)
#     ax.set_ylim(-3, 3)
#     ax.axis('off')

#     # Road layout
#     road_width = 1.0
#     ax.add_patch(plt.Rectangle((-road_width, -3), 2*road_width, 6, color='gray', alpha=0.5))
#     ax.add_patch(plt.Rectangle((-3, -road_width), 6, 2*road_width, color='gray', alpha=0.5))

#     # Signal light
#     light = plt.Circle((0, 0), 0.25, color='green')
#     ax.add_patch(light)

#     # Car storage
#     car_patches = [[] for _ in range(num_lanes)]
#     lane_positions = [(-2.5, 0), (2.5, 0), (0, -2.5), (0, 2.5)]  # L1, L2, L3, L4 directions

#     def draw_cars():
#         """Draw rectangles for cars based on queue lengths"""
#         for lane_cars in car_patches:
#             for car in lane_cars:
#                 car.remove()
#         for i in range(num_lanes):
#             lane_cars = []
#             qlen = min(env.queues[i], 10)  # limit visual queue
#             for j in range(qlen):
#                 if i == 0:  # Left to center
#                     x, y = lane_positions[i][0] + j * 0.25, lane_positions[i][1] - 0.2
#                     rect = plt.Rectangle((x, y), 0.2, 0.4, color='blue')
#                 elif i == 1:  # Right to center
#                     x, y = lane_positions[i][0] - j * 0.25, lane_positions[i][1] + 0.2
#                     rect = plt.Rectangle((x, y), 0.2, 0.4, color='orange')
#                 elif i == 2:  # Bottom to center
#                     x, y = lane_positions[i][0] - 0.2, lane_positions[i][1] + j * 0.25
#                     rect = plt.Rectangle((x, y), 0.4, 0.2, color='green')
#                 else:  # Top to center
#                     x, y = lane_positions[i][0] + 0.2, lane_positions[i][1] - j * 0.25
#                     rect = plt.Rectangle((x, y), 0.4, 0.2, color='red')
#                 ax.add_patch(rect)
#                 lane_cars.append(rect)
#             car_patches[i] = lane_cars

#     draw_cars()

#     title = ax.text(0, 2.7, '', ha='center', fontsize=12, fontweight='bold')

#     def update(frame):
#         """Called every frame by FuncAnimation"""
#         obs = env._get_obs()
#         if agent:
#             action = agent.select_action(obs, eval_mode=True)
#         else:
#             action = np.random.randint(0, env.action_space.n)

#         _, _, done, _, _ = env.step(action)

#         # Update light color
#         if env.phase == 'green':
#             light.set_color('green')
#         elif env.phase == 'yellow':
#             light.set_color('yellow')
#         else:
#             light.set_color('red')

#         # Redraw cars
#         draw_cars()

#         title.set_text(f"Step {env.step_count} | Green Lane: {env.current_green+1} | Phase: {env.phase}")

#         return []

#     ani = FuncAnimation(fig, update, frames=steps, interval=interval, repeat=False)
#     plt.show()




if __name__ == "__main__":
    #Train the agent — will resume from checkpoint if found
    #agent, env = train(num_episodes=17000, save_interval=1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    env = TrafficSignalEnv(num_lanes=4)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create a fresh agent instance

    agent = DQNAgent(obs_dim, action_dim, device=device)

    # Load the latest trained model
    checkpoint = torch.load("dqn_traffic_checkpoint.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    agent.net.load_state_dict(checkpoint["model_state_dict"])
    agent.target.load_state_dict(checkpoint["target_state_dict"])
    agent.net.train()

    # Create an evaluation environment (no penalties for real-world testing)
    yolo_csv_path = r"C:\Users\monee\Desktop\Captsone\data\lane_vehicles.csv"
    df_preview = pd.read_csv(yolo_csv_path)
    num_lanes_from_csv = 4
    eval_env = TrafficSignalEnv(num_lanes=4, min_green=5, yellow_time=2, use_penalties=False)
    

# ✅ Check if file exists before running
    if not os.path.exists(yolo_csv_path):
        raise FileNotFoundError(f"YOLO data file not found at: {yolo_csv_path}")

# ✅ Run evaluation using the CSV file
    evaluate_with_yolo(agent, eval_env, yolo_csv=yolo_csv_path, render=True)

    # def update_with_agent(frame):
    #     obs = env._get_obs()
    #     action = agent.select_action(obs, eval_mode=True)
    #     env.step(action)
    #     env.render()
    # animate_traffic(env, steps=150, interval=400)
    
