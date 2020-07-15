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
    LIST_INCOMING_LANES_LOG_QUEUE_LENGTH =   {
                                'node1': ['0Ato1A_0', '0Ato1A_1', '2Ato1A_0', '2Ato1A_1', 'Nto1A_0', 'Nto1A_1', '1Bto1A_0', '1Bto1A_1'],
                                'node2': ['1Ato2A_0', '1Ato2A_1', '3Ato2A_0', '3Ato2A_1', 'Nto2A_0', 'Nto2A_1', '2Bto2A_0', '2Bto2A_1'],
                                'node3': ['2Ato3A_0', '2Ato3A_1', '4Ato3A_0', '4Ato3A_1', 'Nto3A_0', 'Nto3A_1', '3Bto3A_0', '3Bto3A_1'],
                                'node4': ['3Ato4A_0', '3Ato4A_1', '5Ato4A_0', '5Ato4A_1', 'Nto4A_0', 'Nto4A_1', '4Bto4A_0', '4Bto4A_1'],
                                'node1B': ['0Bto1B_0', '0Bto1B_1', '2Bto1B_0', '2Bto1B_1', '1Ato1B_0', '1Ato1B_1', 'Sto1B_0', 'Sto1B_1'],
                                'node2B': ['1Bto2B_0', '1Bto2B_1', '3Bto2B_0', '3Bto2B_1', '2Ato2B_0', '2Ato2B_1', 'Sto2B_0', 'Sto2B_1'],
                                'node3B': ['2Bto3B_0', '2Bto3B_1', '4Bto3B_0', '4Bto3B_1', '3Ato3B_0', '3Ato3B_1', 'Sto3B_0', 'Sto3B_1'],
                                'node4B': ['3Bto4B_0', '3Bto4B_1', '5Bto4B_0', '5Bto4B_1', '4Ato4B_0', '4Ato4B_1', 'Sto4B_0', 'Sto4B_1']
                            }
elif args.net_file == '4x2-intersections':
    agent_names = ['node1', 'node2', 'node3', 'node4', 'node1B', 'node2B', 'node3B', 'node4B']
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
    LOG_QUEUE_LENGTH_FILE_NAME = './log/%s/heavy-traffic/MARL/queue-length' % args.net_file
    LOG_VEHICLE_FILE_NAME = './log/%s/heavy-traffic/MARL/vehicle' % args.net_file
    LOG_TRAFFIC_LIGHT_FILE_NAME = './log/%s/heavy-traffic/MARL/traffic-light.txt' % args.net_file

elif args.light_traffic:
    LOG_QUEUE_LENGTH_FILE_NAME = './log/%s/light-traffic/MARL/queue-length' % args.net_file
    LOG_VEHICLE_FILE_NAME = './log/%s/light-traffic/MARL/vehicle' % args.net_file
    LOG_TRAFFIC_LIGHT_FILE_NAME = './log/%s/light-traffic/MARL/traffic-light.txt' % args.net_file


# Create multi models, memories of agents
memories = dict()
models = dict()
for agent in agent_names:
    models[agent] = Sequential()
    models[agent].add(Dense(32, input_dim=STATE_SPACE))
    models[agent].add(Activation('relu'))
    models[agent].add(Dense(64))
    models[agent].add(Activation('relu'))
    models[agent].add(Dense(64))
    models[agent].add(Activation('relu'))
    models[agent].add(Dense(ACTION_SPACE))
    models[agent].add(Activation('linear'))
    models[agent].compile(loss='mean_squared_error', optimizer='adam')
    # models[agent].summary()

    memories[agent] = Memory(MEMORY_SIZE)

if args.trial:
    for agent in agent_names:
        if args.heavy_traffic:
            name_file = "./model/%s/heavy-traffic/%s.h5" % (args.net_file, agent)
        elif args.light_traffic:
            name_file = "model/%s/light-traffic/%s.h5" % (args.net_file, agent)
        models[agent].load_weights(name_file)
    env.reset()
    while True:
        actions = dict()
        for agent in agent_names:
            state = env.get_observation(agent)
            q_values = models[agent].predict(np.reshape([state], [1, STATE_SPACE]))
            action = np.argmax(q_values)
            actions[agent] = action
        _, _, is_finished = env.set_action(actions)
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

        for agent in agent_names:
            memories[agent].add([states[agent], actions[agent], rewards[agent], next_states[agent], is_finished])
        if is_finished:
            break

# function to update Q learning
def update_model():
    for agent in agent_names:
        minibatch =  memories[agent].sample(BATCH_SIZE)
        batch_states = []
        batch_targets = []
        for state, action, reward, next_state, done in minibatch:
            batch_states.append(state)
            target = reward
            if not done:
                qs = models[agent].predict(np.reshape([next_state], [1, STATE_SPACE]))
                target = (reward +  GAMMA * np.amax(qs[0]))
            target_f = models[agent].predict(np.reshape([state], [1, STATE_SPACE]))
            target_f[0][action] = target
            batch_targets.append(target_f[0])
        models[agent].fit(np.array(batch_states), np.array(batch_targets), epochs=EPOCHS, shuffle=False, verbose=0, validation_split=0.3)

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
        for agent in agent_names:
            states[agent] = env.get_observation(agent)
            if np.random.rand() <= EPSILON:
                action = random.randint(0, 1)
            else:
                state = np.reshape([states[agent]], [1, STATE_SPACE])
                q_values = models[agent].predict(state)
                action = np.argmax(q_values)
            actions[agent] = action
        rewards, next_states, is_finished = env.set_action(actions)

        for agent in agent_names:
            memories[agent].add([states[agent], actions[agent], rewards[agent], next_states[agent], is_finished])
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
        for agent in agent_names:
            if args.heavy_traffic:
                name_file = "./model/%s/heavy-traffic/%s.h5" % (args.net_file, agent)
            elif args.light_traffic:
                name_file = "model/%s/light-traffic/%s.h5" % (args.net_file, agent)
            os.makedirs(os.path.dirname(name_file), exist_ok=True)
            models[agent].save_weights(name_file)

# memory = SequentialMemory(limit=50000, window_length=1)
# policy = BoltzmannQPolicy()
# dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
#                target_model_update=1e-2, policy=policy)
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# # Okay, now it's time to learn something! We visualize the training here for show, but this
# # slows down training quite a lot. You can always safely abort the training prematurely using
# # Ctrl + C.
# dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

# # After training is done, we save the final weights.
# dqn.save_weights('dqn_{}_weights.h5f'.format(args.net_file), overwrite=True)

# # Finally, evaluate our algorithm for 5 episodes.
# dqn.test(env, nb_episodes=5, visualize=True)
