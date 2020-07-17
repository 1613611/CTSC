import numpy as np
import tensorflow as tf
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from SumoEnvironment import Environment
from collections import deque
from Memory import Memory
import random
import argparse
from prettytable import PrettyTable
import os

ACTION_SPACE = 2
STATE_SPACE = 4
N_AGENTS = 4

parser = argparse.ArgumentParser() 
parser.add_argument("--gui", action='store_true', default=False, help = "Enable gui")
parser.add_argument('--test-site', type=str, dest="net_file", help='Name of the net file')
parser.add_argument("--light-traffic", action='store_true', dest="light_traffic", default=False, help = "Use workload of light traffic") 
parser.add_argument("--heavy-traffic", action='store_true', dest="heavy_traffic", default=False, help = "Use workload of heavy traffic")
parser.add_argument('--step-size', type=int, dest="step_size", default=5, help='Value of the step size')
parser.add_argument('--number-episodes-train', type=int, dest="n_episodes", default=1000)
parser.add_argument('--number-episodes-pretrain', type=int, dest="n_episodes_pretrain", default=5)
parser.add_argument('--random-seed', type=int, dest="random_seed", default=42)
parser.add_argument('--memory-length', type=int, dest="memory_length", default=4192)
parser.add_argument('--batch-size', type=int, dest="batch_size", default=512)
parser.add_argument('--epsilon', type=float, dest="epsilon", default=0.05)
parser.add_argument('--update-interval', type=int, dest="update_interval", default=300)
parser.add_argument('--epochs', type=int, dest="epochs", default=50)
parser.add_argument('--gamma', type=float, dest="gamma", default=0.95)
parser.add_argument('--max-step', type=int, dest="max_step", default=7200)
parser.add_argument('--trial', action='store_true', dest="trial", default=False)
parser.add_argument('--train', action='store_true', dest="train", default=False)

args = parser.parse_args()
# set RANDOM SEED
RANDOM_SEED = args.random_seed
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED']=str(RANDOM_SEED)
tf.random.set_random_seed(RANDOM_SEED)

N_EPISODES_TRAIN = args.n_episodes
N_EPISODES_PRETRAIN = args.n_episodes_pretrain
MEMORY_SIZE = args.memory_length
BATCH_SIZE = args.batch_size
EPSILON = args.epsilon
UPDATE_INTERVAL = int(args.update_interval / args.step_size)
EPOCHS = args.epochs
GAMMA = args.gamma

# Creat the environment 
env = Environment(args)

if args.net_file == '4x1-two-way':
    agent_names = ['node1', 'node2', 'node3', 'node4']
    N_AGENTS = 4
    LIST_INCOMING_LANES_LOG_QUEUE_LENGTH = [ '0to1_0', 'NtoC_1_0', '2to1_0', 'StoC_1_0', '1to2_0', 'NtoC_2_0', '3to2_0', 'StoC_2_0',\
            '2to3_0', 'NtoC_3_0', '4to3_0', 'StoC_3_0', '3to4_0', 'NtoC_4_0', '5to4_0', 'StoC_4_0',\
            '0to1_1', 'NtoC_1_1', '2to1_1', 'StoC_1_1', '1to2_1', 'NtoC_2_1', '3to2_1', 'StoC_2_1',\
            '2to3_1', 'NtoC_3_1', '4to3_1', 'StoC_3_1', '3to4_1', 'NtoC_4_1', '5to4_1', 'StoC_4_1']
elif args.net_file == '4x1-one-way':
    agent_names = ['node1', 'node2', 'node3', 'node4']
    N_AGENTS = 4
    LIST_INCOMING_LANES_LOG_QUEUE_LENGTH = ['0to1_0', '0to1_1', '0to1_2', '0to1_3', 'NtoC_1_0', 'NtoC_1_1', 'StoC_1_0', 'StoC_1_1', \
                                    '1to2_0', '1to2_1', '1to2_2', '1to2_3', 'NtoC_2_0', 'NtoC_2_1', 'StoC_2_0', 'StoC_2_1', \
                                    '2to3_0', '2to3_1', '2to3_2', '2to3_3', 'NtoC_3_0', 'NtoC_3_1', 'StoC_3_0', 'StoC_3_1', \
                                    '3to4_0', '3to4_1', '3to4_2', '3to4_3', 'NtoC_4_0', 'NtoC_4_1', 'StoC_4_0', 'StoC_4_1']
elif args.net_file == '4x2-intersections':
    agent_names = ['node1', 'node2', 'node3', 'node4', 'node1B', 'node2B', 'node3B', 'node4B']
    N_AGENTS = 8
    LIST_INCOMING_LANES_LOG_QUEUE_LENGTH = [ '0Ato1A_0', '0Ato1A_1', '2Ato1A_0', '2Ato1A_1', 'Nto1A_0', 'Nto1A_1', '1Bto1A_0', '1Bto1A_1',\
                        '1Ato2A_0', '1Ato2A_1', '3Ato2A_0', '3Ato2A_1', 'Nto2A_0', 'Nto2A_1', '2Bto2A_0', '2Bto2A_1',\
                        '2Ato3A_0', '2Ato3A_1', '4Ato3A_0', '4Ato3A_1', 'Nto3A_0', 'Nto3A_1', '3Bto3A_0', '3Bto3A_1',\
                        '3Ato4A_0', '3Ato4A_1', '5Ato4A_0', '5Ato4A_1', 'Nto4A_0', 'Nto4A_1', '4Bto4A_0', '4Bto4A_1',\
                        '0Bto1B_0', '0Bto1B_1', '2Bto1B_0', '2Bto1B_1', '1Ato1B_0', '1Ato1B_1', 'Sto1B_0', 'Sto1B_1',\
                        '1Bto2B_0', '1Bto2B_1', '3Bto2B_0', '3Bto2B_1', '2Ato2B_0', '2Ato2B_1', 'Sto2B_0', 'Sto2B_1',\
                        '2Bto3B_0', '2Bto3B_1', '4Bto3B_0', '4Bto3B_1', '3Ato3B_0', '3Ato3B_1', 'Sto3B_0', 'Sto3B_1',\
                        '3Bto4B_0', '3Bto4B_1', '5Bto4B_0', '5Bto4B_1', '4Ato4B_0', '4Ato4B_1', 'Sto4B_0', 'Sto4B_1',\
                    ]


if args.heavy_traffic:
    LOG_QUEUE_LENGTH_FILE_NAME = './log/%s/heavy-traffic/CentralizedRL/queue-length' % args.net_file
    LOG_VEHICLE_FILE_NAME = './log/%s/heavy-traffic/CentralizedRL/vehicle' % args.net_file
    LOG_TRAFFIC_LIGHT_FILE_NAME = './log/%s/heavy-traffic/CentralizedRL/traffic-light.txt' % args.net_file

elif args.light_traffic:
    LOG_QUEUE_LENGTH_FILE_NAME = './log/%s/light-traffic/CentralizedRL/queue-length' % args.net_file
    LOG_VEHICLE_FILE_NAME = './log/%s/light-traffic/CentralizedRL/vehicle' % args.net_file
    LOG_TRAFFIC_LIGHT_FILE_NAME = './log/%s/light-traffic/CentralizedRL/traffic-light.txt' % args.net_file

# Create multi models, memories of agents

model = Sequential()
model.add(Dense(32, input_dim=STATE_SPACE*N_AGENTS))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(ACTION_SPACE*N_AGENTS))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

memory = Memory(MEMORY_SIZE)

if args.trial:
    if args.heavy_traffic:
        name_file = "./model/%s/heavy-traffic/Centralized/model.h5" % args.net_file
    elif args.light_traffic:
        name_file = "model/%s/light-traffic/Centralized/model.h5" % args.net_file
    model.load_weights(name_file)
    env.reset()

    while True:
        states = dict()
        actions = dict()
        rewards = dict()
        next_states = dict()

        step_action = []
        
        np_state = []
        for agent in agent_names:
            states[agent] = env.get_observation(agent)
            np_state.extend(states[agent])        
        q_values = model.predict(np.reshape([np_state], [1, STATE_SPACE*N_AGENTS]))
        for idx, agent in enumerate(agent_names):
            actions[agent] = np.argmax(q_values[0][idx*2:idx*2+2])                
        rewards, next_states, is_finished = env.set_action(actions)
        if is_finished:
            break
    log_QL, log_Veh, tls_log = env.get_log()
    t = PrettyTable(['Feature', 'Value'])
    t.add_row(['Average Queue Length', np.mean(log_QL)])
    t.add_row(['Average Travel Time', np.mean([veh['travel_time'] for _, veh in log_Veh.items()])])
    t.add_row(['Average Speed', np.mean([veh['average_speed'] for _, veh in log_Veh.items()])])  
    print(t)

    os.makedirs(os.path.dirname(LOG_TRAFFIC_LIGHT_FILE_NAME), exist_ok=True)
    log_tls_file = open(LOG_TRAFFIC_LIGHT_FILE_NAME, "w")
    log_tls_file.write('step')
    for agent in agent_names:
        log_tls_file.write(',%s' % agent)
    log_tls_file.write('\n')
    for idx, log in enumerate(tls_log):
        log_tls_file.write('%d' % idx)
        [log_tls_file.write(',%d' % val) for val in log]
        log_tls_file.write('\n')
    sys.exit(0)

    

for _ in range(N_EPISODES_PRETRAIN):
    env.reset()
    while True:
        states = dict()
        actions = dict()
        rewards = dict()
        next_states = dict()

        step_action = []
        for agent in agent_names:
            states[agent] = env.get_observation(agent)
            action = random.randint(0, 1)
            actions[agent] = action
        rewards, next_states, is_finished = env.set_action(actions)

        memory.add([states, actions, rewards, next_states, is_finished])
        if is_finished:
            break

# function to update Q learning
def update_model():
    minibatch =  memory.sample(BATCH_SIZE)    
    batch_states = []
    batch_targets = []
    
    for states, actions, rewards, next_states, done in minibatch:
        np_states = []
        np_actions = []
        np_rewards = []
        np_next_states = []
        for agent in agent_names:
            np_states.extend(states[agent])
            np_actions.append(actions[agent])
            np_rewards.append(rewards[agent])
            np_next_states.extend(next_states[agent])

        target = np_rewards
        if not done:
            qs = model.predict(np.reshape([np_next_states], [1, STATE_SPACE*N_AGENTS]))
            for idx,_ in enumerate(np_actions):
                target[idx] = (np_rewards[idx] +  GAMMA * np.amax(qs[0][idx*2:idx*2+2]))

        target_f = model.predict(np.reshape([np_states], [1, STATE_SPACE*N_AGENTS]))
        for idx, action in enumerate(np_actions):
            target_f[0][idx*2 + action] = target[idx]
        
        batch_targets.append(target_f[0])
        batch_states.append(np_states)
    model.fit(np.array(batch_states), np.array(batch_targets), epochs=EPOCHS, shuffle=False, verbose=0, validation_split=0.3)


# train
for epi in range(N_EPISODES_TRAIN):
    env.reset()
    count_interval = UPDATE_INTERVAL
    while True:
        states = dict()
        actions = dict()
        rewards = dict()
        next_states = dict()

        step_action = []
        
        np_state = []
        for agent in agent_names:
            states[agent] = env.get_observation(agent)
            np_state.extend(states[agent])
        
        if np.random.rand() <= EPSILON:
            for agent in agent_names:
                actions[agent] = random.randint(0, 1)
        else:
            q_values = model.predict(np.reshape([np_state], [1, STATE_SPACE*N_AGENTS]))
            for idx, agent in enumerate(agent_names):
                actions[agent] = np.argmax(q_values[0][idx*2:idx*2+2])                
        rewards, next_states, is_finished = env.set_action(actions)

        memory.add([states, actions, rewards, next_states, is_finished])
        if is_finished:
            break

        # count interval to update
        count_interval -= 1
        if count_interval <= 0:
            count_interval = UPDATE_INTERVAL
            update_model()

    #finish => update
    update_model()
    
    # Log to visualize
    log_QL, log_Veh, _ = env.get_log()
    log_QL_address = LOG_QUEUE_LENGTH_FILE_NAME + '/%d.txt' % epi
    os.makedirs(os.path.dirname(log_QL_address), exist_ok=True)
    log_QL_file = open(log_QL_address, "w")
    log_QL_file.write('step')
    [log_QL_file.write(',%s' % lane) for lane in LIST_INCOMING_LANES_LOG_QUEUE_LENGTH]
    log_QL_file.write('\n')
    for idx, q_length in enumerate(log_QL):
        str_q_length = str(idx)
        for q in q_length:
            str_q_length += ',%d' % q
        log_QL_file.write(str_q_length + '\n')

    log_Veh_address = LOG_VEHICLE_FILE_NAME + '/%d.txt' % epi
    os.makedirs(os.path.dirname(log_Veh_address), exist_ok=True)
    log_Veh_file = open(log_Veh_address, "w")
    log_Veh_file.write('VehID,DepartedTime,ArrivedTime,RouteID,TravelTime,AverageSpeed\n')            

    for vehicleId in log_Veh.keys():
        log_Veh_file.write('%s,%f,%f,%s,%f,%f\n' % (vehicleId,  log_Veh[vehicleId]['departed'],\
                                                                log_Veh[vehicleId]['arrived'],\
                                                                log_Veh[vehicleId]['routeID'],\
                                                                log_Veh[vehicleId]['travel_time'],\
                                                                log_Veh[vehicleId]['average_speed']))

    t = PrettyTable(['Feature', 'Value'])
    t.add_row(['Episode', epi])
    t.add_row(['Average Queue Length', np.mean(log_QL)])
    t.add_row(['Average Travel Time', np.mean([veh['travel_time'] for _, veh in log_Veh.items()])])
    t.add_row(['Average Speed', np.mean([veh['average_speed'] for _, veh in log_Veh.items()])])  
    print(t)

    if args.train == True:
        if args.heavy_traffic:
            name_file = "./model/%s/heavy-traffic/Centralized/model.h5" % args.net_file
        elif args.light_traffic:
            name_file = "model/%s/light-traffic/Centralized/model.h5" % args.net_file
        os.makedirs(os.path.dirname(name_file), exist_ok=True)
        model.save_weights(name_file)

