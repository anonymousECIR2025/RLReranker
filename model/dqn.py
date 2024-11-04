import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torchcontrib.optim import SWA
from collections import deque
from util.preprocess import *
from util.helper_functions import set_manual_seed

set_manual_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, model_size):
        super(DQN, self).__init__()
        self.input_dim = input_dim + 1 
        self.output_dim = output_dim

        layer_sizes = {
            "head": [],
            "small": [32],
            "medium": [192, 48, 24],
            "big": [384, 192, 96, 48, 24],
            "large": [512, 384, 256, 192, 128, 64, 32],
            "xlarge": [640, 580, 512, 384, 256, 192, 128, 64, 32],
            "xxlarge": [698, 640, 580, 512, 384, 256, 192, 128, 64, 32, 16, 8],
            "xxxlarge": [698, 640, 620, 580, 560, 512, 384, 256, 192, 128, 64, 32, 16, 8],
            "xxxxlarge": [698, 640, 620, 600, 580, 560, 540, 512, 384, 256, 192, 128, 64, 32, 16, 8],
            "xxxxxlarge": [698, 640, 620, 600, 580, 560, 540, 512, 384, 360, 340, 300, 256, 192, 128, 64, 32, 16, 8],
        }.get(model_size, [32])

        layers = []
        if layer_sizes:

            layers.append(nn.Linear(self.input_dim, layer_sizes[0]))
            layers.append(nn.ReLU())
            for i in range(len(layer_sizes) - 1):
                layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.ReLU()]
            layers.append(nn.Linear(layer_sizes[-1], self.output_dim))
        else:
            layers.append(nn.Linear(self.input_dim, self.output_dim))

        self.fc = nn.Sequential(*layers)
        self.to(device)

    def forward(self, state):
        state = state.to(device)
        return self.fc(state)

    
class DQNAgent:

    def __init__(self, dataset, buffer, config_dict, pre_trained_model=None):

        self.dataset = dataset 
        self.replay_buffer = buffer

        if config_dict.features:
            print("----------")
            self.features = list(config_dict.features)
            len_features = len(self.features)
        else:
            print("+++++++++")
            self.features = None
            len_features = 0

        print(self.features)

        self.model = (pre_trained_model or DQN(
            config_dict.get('input_dim', 768) + len_features, 
            config_dict.get('output_dim', 1), 
            config_dict.get('model_size', 'small')
        )).to(device)

        self.target_model = (pre_trained_model or DQN(
            config_dict.get('input_dim', 768) + len_features, 
            config_dict.get('output_dim', 1), 
            config_dict.get('model_size', 'small')
        )).to(device)

        self.learning_rate = config_dict.get('learning_rate', 1e-4)
        self.gamma = config_dict.get('gamma', 0.99)
        self.tau = config_dict.get('tau', 0.005)
        self.swa = config_dict.get('swa', False)

        self.MSE_loss = nn.MSELoss().to(device)

        base_opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.swa:
            self.optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
        else:
            self.optimizer = base_opt


    def get_action(self, state, dataset=None):

        dataset = dataset if not dataset.empty else self.dataset

        inputs, relevance_list = get_multiple_model_inputs(state, state.remaining, self.features, dataset)
        
        model_inputs = autograd.Variable(torch.from_numpy(inputs).float().unsqueeze(0)).to(device)
        
        expected_returns = self.model.forward(model_inputs)
        expected_returns_np = expected_returns.detach().cpu().numpy()

        max_relevance_index = np.argmax(relevance_list)
        max_expected_index = np.argmax(expected_returns_np)

        valid_expected = (max_relevance_index == max_expected_index)
        value, index = expected_returns.max(1)
        return state.remaining[max_expected_index], valid_expected


    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def compute_loss(self, batch, dataset, normalized):

        states, actions, rewards, next_states, dones = batch

        model_inputs = torch.tensor([get_model_inputs(states[i], actions[i], self.features, dataset, normalized) \
                                for i in range(len(states))], dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        curr_Q = self.model(model_inputs)


        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        next_model_inputs = torch.tensor([get_model_inputs(next_states[i], actions[i], self.features, dataset, normalized) \
                                    for i in range(len(next_states))], dtype=torch.float32, device=device)

        next_Q = self.target_model.forward(model_inputs)
        next_Q = self.model(next_model_inputs)
        max_next_Q = torch.max(next_Q, dim=1)[0]
        expected_Q = rewards.squeeze(1) + (1 - dones) * self.gamma * max_next_
        
        loss = self.MSE_loss(curr_Q.squeeze(0), expected_Q.detach())
        return loss, curr_Q, expected_Q

    def update(self, batch_size, normalized):
        batch = self.replay_buffer.sample(batch_size)

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        loss, curr_Q, expected_Q = self.compute_loss(batch, self.dataset, normalized)

        train_loss = loss.float()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.swa:
            self.optimizer.swap_swa_sgd()
        return train_loss, curr_Q, expected_Q,  reward_batch, done_batch