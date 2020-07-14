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

if args.net_file == '4-arterial-intersections':
    agent_names = ['node1', 'node2', 'node3', 'node4']
    N_AGENTS = 4
if args.heavy_traffic:
    LOG_QUEUE_LENGTH_FILE_NAME = './log/%s/heavy-traffic/CentralizedRLMapping/queue-length' % args.net_file
    LOG_VEHICLE_FILE_NAME = './log/%s/heavy-traffic/CentralizedRLMapping/vehicle' % args.net_file

    LOG_TRAFFIC_LIGHT_FILE_NAME = './log/%s/heavy-traffic/CentralizedRLMapping/traffic-light.txt' % args.net_file

elif args.light_traffic:
    LOG_QUEUE_LENGTH_FILE_NAME = './log/%s/light-traffic/CentralizedRLMapping/queue-length' % args.net_file
    LOG_VEHICLE_FILE_NAME = './log/%s/light-traffic/CentralizedRLMapping/vehicle' % args.net_file

    LOG_TRAFFIC_LIGHT_FILE_NAME = './log/%s/light-traffic/CentralizedRLMapping/traffic-light.txt' % args.net_file

# Create multi models, memories of agents

model = Sequential()
model.add(Dense(32, input_dim=STATE_SPACE*N_AGENTS))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(ACTION_SPACE**N_AGENTS))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

memory = Memory(MEMORY_SIZE)

if args.trial:
    if args.heavy_traffic:
        name_file = "./model/%s/heavy-traffic/CentralizedRLMapping/model.h5" % args.net_file
    elif args.light_traffic:
        name_file = "model/%s/light-traffic/CentralizedRLMapping/model.h5" % args.net_file
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
        action_map = np.argmax(q_values[0])
        action_bin = bin(action_map)[2:]
        for idx, agent in enumerate(agent_names):
            if idx >= len(action_bin):
                actions[agent] = 0
            else:  
                actions[agent] = int(action_bin[idx])
        
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
    log_tls_file.write('step,node1,node2,node3,node4\n')
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

        target = np.sum(np_rewards)
        if not done:
            qs = model.predict(np.reshape([np_next_states], [1, STATE_SPACE*N_AGENTS]))
            action_bin = ''
            for action in np_actions:
                action_bin += str(action)
            action_map = int(action_bin, 2)
            target =  np.sum(np_rewards) + GAMMA * np.max(qs[0])

        target_f = model.predict(np.reshape([np_states], [1, STATE_SPACE*N_AGENTS]))
        target_f[0][action_map] = target
        
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
            action_map = np.argmax(q_values[0])
            action_bin = bin(action_map)[2:]
            for idx, agent in enumerate(agent_names):
                if idx >= len(action_bin):
                    actions[agent] = 0
                else:  
                    actions[agent] = int(action_bin[idx])
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
    if args.net_file == '4-arterial-intersections':
        log_QL_file.write('STEP,W11,W12,N11,N12,E11,E12,S11,S12,W21,W22,N21,N22,E21,E22,S21,S22,W31,W32,N31,N32,E31,E32,S31,S32,W41,W42,N41,N42,E41,E42,S41,S42\n')
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
            name_file = "./model/%s/heavy-traffic/CentralizedRLMapping/model.h5" % args.net_file
        elif args.light_traffic:
            name_file = "model/%s/light-traffic/CentralizedRLMapping/model.h5" % args.net_file
        os.makedirs(os.path.dirname(name_file), exist_ok=True)
        model.save_weights(name_file)