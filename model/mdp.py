import numpy as np
import pandas as pd
import random
from collections import deque
from loguru import logger
from util.preprocess import *
from util.helper_functions import set_manual_seed

set_manual_seed()

def compute_reward(t, features_coefs, sorted_list=None, reward_mode="_"):

    relevance, relevance_coef = features_coefs["relevance"][0], features_coefs["relevance"][1]
    bias, bias_coef = features_coefs["bias"][0], features_coefs["bias"][1]

    if reward_mode =="relevance":
        reward = (relevance * relevance_coef)
        return reward

    elif reward_mode == "relevance_t":
        return (relevance * relevance_coef)/ t

    elif reward_mode == "zero_one":
        if relevance >= 1:
            return  1
        else:
            return  0
    else:
        return (relevance * relevance_coef)
    
class State:

    def __init__(self, t, query, remaining, sorted_list=None):

        self.t = t
        self.qid = query
        self.remaining = remaining
        self.sorted_list = sorted_list or []

    def pop(self):
        return self.remaining.pop()

    def initial(self):
        return self.t == 0

    def terminal(self):
        return len(self.remaining) == 0

    def __str__(self):
        return f"t:{self.t}, qid:{self.qid}, sorted_list:{self.sorted_list}"

class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def push_batch(self, qid_list_path, df, reward_params, n):

        reward_mode = reward_params["reward_mode"]
        relevance_coef = reward_params["relevance_coef"]

        for i in range(n):

            random_qid = random.choice(list(df["qid"]))
            with open(qid_list_path, 'a+') as f:
                f.write(f"{random_qid}\n")

            filtered_df = df.loc[df["qid"] == str(random_qid)].reset_index()

            row_order = [x for x in range(len(filtered_df))]
            X = [x[1]["doc_id"] for x in filtered_df.iterrows()]
            sorted_list = []

            random.shuffle(row_order)
            for t,r in enumerate(row_order): 
                cur_row = filtered_df.iloc[r]
                old_state = State(t, cur_row["qid"], X[:], sorted_list[:])
                action = cur_row["doc_id"]
                X.remove(action)
                sorted_list.append([action, cur_row["relevance"]])

                new_state = State(t+1, cur_row["qid"], X[:], sorted_list[:])

                relevance_score = cur_row.get("relevance", 0)

                features_coefs = {
                                "relevance": (relevance_score, relevance_coef) }
                reward = compute_reward(t+1, features_coefs, sorted_list[:], reward_mode)
                self.push(old_state, action, reward, new_state, t+1 == len(row_order))
                filtered_df.drop(filtered_df.index[[r]])


    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)