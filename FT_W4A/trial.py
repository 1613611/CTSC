import tensorflow as tf
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import json
import os
import glob
import urllib.request as urllib
from SumoEnvironment import SumoEnvironment
path = os.getcwd()
import requests 
import pickle
import time
def read_params():
    with open('./params.json') as f:
        return json.load(f)

class Trial():
    def trial(self):
        ts = time.time()
        game = SumoEnvironment()
        current_phase, state = game.reset()
        while True:
            # Qs = sess.run(self.output, feed_dict = {
            #                                             self.inputs_: state.reshape(1, state.shape[0]),
            #                                             self.current_phases_: np.asarray([float(current_phase)]),    
            #                                         })
            # choice = np.argmax(Qs)
            choice  = random.randint(0, 1)
            current_phase, state, _, done, _ = game.step(choice)
            if done:
                now = time.time()
                log_reward = game.get_log_reward()
                # print(log_reward['duration'])
                print(np.mean(log_reward['q_length']))
                print(np.mean(log_reward['delay']))
                # print(np.mean(log_reward['duration']))
                # print(np.mean(q_length_arr))
                # print(np.mean(delay_arr))
                # print(np.mean(waiting_arr))
                # print(np.mean(travel_arr))
                # print(game.get_total_reward())
                print(game.get_duration())
                print(now - ts)
                pickle.dump(np.asarray([np.mean(log_reward['q_length']), np.mean(log_reward['delay']), game.get_duration(), now - ts]), open('./log_trial','wb'))
                break

trial = Trial()
trial.trial()