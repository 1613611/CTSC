import os
import sys
import argparse
import numpy as np
from prettytable import PrettyTable
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci


parser = argparse.ArgumentParser() 
parser.add_argument("--gui", action='store_true', help = "Enable gui")
parser.add_argument('--test-site', type=str, dest="net_file", help='Name of the net file')
parser.add_argument("--light-traffic", action='store_true', dest="light_traffic", default=False, help = "Use workload of light traffic") 
parser.add_argument("--heavy-traffic", action='store_true', dest="heavy_traffic", default=False, help = "Use workload of heavy traffic") 
parser.add_argument('--max-step', type=int, dest="max_step", default=7200)

args = parser.parse_args()

if args.gui:
    sumoBinary = "/usr/bin/sumo-gui"
else:
    sumoBinary = "/usr/bin/sumo"
sumoConfig = ["-c", "./network/network.sumocfg"]
sumoCmd = [sumoBinary, sumoConfig[0], sumoConfig[1]]

sumoCmd.extend(['-n', './network/%s.net.xml' % args.net_file])

if args.net_file == '4-arterial-intersections':
    LIST_INCOMING_LANES =   {
                                'node1': ['0to1_0', '0to1_1', '2to1_0', '2to1_1', 'NtoC_1_0', 'NtoC_1_1', 'StoC_1_0', 'StoC_1_1'],
                                'node2': ['1to2_0', '1to2_1', '3to2_0', '3to2_1', 'NtoC_2_0', 'NtoC_2_1', 'StoC_2_0', 'StoC_2_1'],
                                'node3': ['2to3_0', '2to3_1', '4to3_0', '4to3_1', 'NtoC_3_0', 'NtoC_3_1', 'StoC_3_0', 'StoC_3_1'],
                                'node4': ['3to4_0', '3to4_1', '5to4_0', '5to4_1', 'NtoC_4_0', 'NtoC_4_1', 'StoC_4_0', 'StoC_4_1']
                            }
    LIST_OUTGOING_LANES =   {
                                'node1': ['1to2_0', '1to2_1', '1to0_0', '1to0_1', 'CtoS_1_0', 'CtoS_1_1', 'CtoN_1_0', 'CtoN_1_1'],
                                'node2': ['2to3_0', '2to3_1', '2to1_0', '2to1_1', 'CtoS_2_0', 'CtoS_2_1', 'CtoN_2_0', 'CtoN_2_1'],
                                'node3': ['3to4_0', '3to4_1', '3to2_0', '3to2_1', 'CtoS_3_0', 'CtoS_3_1', 'CtoN_3_0', 'CtoN_3_1'],
                                'node4': ['4to5_0', '4to5_1', '4to3_0', '4to3_1', 'CtoS_4_0', 'CtoS_4_1', 'CtoN_4_0', 'CtoN_4_1'],
                            }
    LIST_INCOMING_LANES_LOG_QUEUE_LENGTH = [ '0to1_0', 'NtoC_1_0', '2to1_0', 'StoC_1_0', '1to2_0', 'NtoC_2_0', '3to2_0', 'StoC_2_0',\
                '2to3_0', 'NtoC_3_0', '4to3_0', 'StoC_3_0', '3to4_0', 'NtoC_4_0', '5to4_0', 'StoC_4_0',\
                '0to1_1', 'NtoC_1_1', '2to1_1', 'StoC_1_1', '1to2_1', 'NtoC_2_1', '3to2_1', 'StoC_2_1',\
                '2to3_1', 'NtoC_3_1', '4to3_1', 'StoC_3_1', '3to4_1', 'NtoC_4_1', '5to4_1', 'StoC_4_1']

    DISTANCE_OF_ROUTE = {"route1": 830, "route2": 830, "route1A": 320, "route1B": 320, "route2A": 320, "route2B": 320, \
                         "route3A": 320, "route3B": 320, "route4A": 320, "route4B": 320}    
    TRAFFIC_SIGNAL_LIGHT_NAMES = ['node1', 'node2', 'node3', 'node4']
elif args.net_file == '4x2-intersections':
    LIST_INCOMING_LANES =   {
                            'node1': ['0Ato1A_0', '0Ato1A_1', '2Ato1A_0', '2Ato1A_1', 'Nto1A_0', 'Nto1A_1', '1Bto1A_0', '1Bto1A_1'],
                            'node2': ['1Ato2A_0', '1Ato2A_1', '3Ato2A_0', '3Ato2A_1', 'Nto2A_0', 'Nto2A_1', '2Bto2A_0', '2Bto2A_1'],
                            'node3': ['2Ato3A_0', '2Ato3A_1', '4Ato3A_0', '4Ato3A_1', 'Nto3A_0', 'Nto3A_1', '3Bto3A_0', '3Bto3A_1'],
                            'node4': ['3Ato4A_0', '3Ato4A_1', '5Ato4A_0', '5Ato4A_1', 'Nto4A_0', 'Nto4A_1', '4Bto4A_0', '4Bto4A_1'],
                            'node1B': ['0Bto1B_0', '0Bto1B_1', '2Bto1B_0', '2Bto1B_1', '1Ato1B_0', '1Ato1B_1', 'Sto1B_0', 'Sto1B_1'],
                            'node2B': ['1Bto2B_0', '1Bto2B_1', '3Bto2B_0', '3Bto2B_1', '2Ato2B_0', '2Ato2B_1', 'Sto2B_0', 'Sto2B_1'],
                            'node3B': ['2Bto3B_0', '2Bto3B_1', '4Bto3B_0', '4Bto3B_1', '3Ato3B_0', '3Ato3B_1', 'Sto3B_0', 'Sto3B_1'],
                            'node4B': ['3Bto4B_0', '3Bto4B_1', '5Bto4B_0', '5Bto4B_1', '4Ato4B_0', '4Ato4B_1', 'Sto4B_0', 'Sto4B_1']
                        }
    LIST_OUTGOING_LANES =   {
                            'node1': ['1Ato2A_0', '1Ato2A_1', '1Ato0A_0', '1Ato0A_1', '1Ato1B_0', '1Ato1B_1', '1AtoN_0', '1AtoN_1'],
                            'node2': ['2Ato3A_0', '2Ato3A_1', '2Ato1A_0', '2Ato1A_1', '2Ato2B_0', '2Ato2B_1', '2AtoN_0', '2AtoN_1'],
                            'node3': ['3Ato4A_0', '3Ato4A_1', '3Ato2A_0', '3Ato2A_1', '3Ato3B_0', '3Ato3B_1', '3AtoN_0', '3AtoN_1'],
                            'node4': ['4Ato5A_0', '4Ato5A_1', '4Ato3A_0', '4Ato3A_1', '4Ato4B_0', '3Ato3B_1', '4AtoN_0', '4AtoN_1'],
                            'node1B': ['1Bto2B_0', '1Bto2B_1', '1Bto0B_0', '1Bto0B_1', '1BtoS_0', '1BtoS_1', '1Bto1A_0', '1Bto1A_1'],                            
                            'node2B': ['2Bto3B_0', '2Bto3B_1', '2Bto1B_0', '2Bto1B_1', '2BtoS_0', '2BtoS_1', '2Bto2A_0', '2Bto2A_1'],                             
                            'node3B': ['3Bto4B_0', '3Bto4B_1', '3Bto2B_0', '3Bto2B_1', '3BtoS_0', '3BtoS_1', '3Bto3A_0', '3Bto3A_1'],
                            'node4B': ['4Bto5B_0', '4Bto5B_1', '4Bto3B_0', '4Bto3B_1', '4BtoS_0', '4BtoS_1', '4Bto4A_0', '4Bto4A_1'],                                                                                                               
                        }

    LIST_INCOMING_LANES_LOG_QUEUE_LENGTH = [ '0Ato1A_0', '0Ato1A_1', '2Ato1A_0', '2Ato1A_1', 'Nto1A_0', 'Nto1A_1', '1Bto1A_0', '1Bto1A_1',\
                            '1Ato2A_0', '1Ato2A_1', '3Ato2A_0', '3Ato2A_1', 'Nto2A_0', 'Nto2A_1', '2Bto2A_0', '2Bto2A_1',\
                            '2Ato3A_0', '2Ato3A_1', '4Ato3A_0', '4Ato3A_1', 'Nto3A_0', 'Nto3A_1', '3Bto3A_0', '3Bto3A_1',\
                            '3Ato4A_0', '3Ato4A_1', '5Ato4A_0', '5Ato4A_1', 'Nto4A_0', 'Nto4A_1', '4Bto4A_0', '4Bto4A_1',\
                            '0Bto1B_0', '0Bto1B_1', '2Bto1B_0', '2Bto1B_1', '1Ato1B_0', '1Ato1B_1', 'Sto1B_0', 'Sto1B_1',\
                            '1Bto2B_0', '1Bto2B_1', '3Bto2B_0', '3Bto2B_1', '2Ato2B_0', '2Ato2B_1', 'Sto2B_0', 'Sto2B_1',\
                            '2Bto3B_0', '2Bto3B_1', '4Bto3B_0', '4Bto3B_1', '3Ato3B_0', '3Ato3B_1', 'Sto3B_0', 'Sto3B_1',\
                            '3Bto4B_0', '3Bto4B_1', '5Bto4B_0', '5Bto4B_1', '4Ato4B_0', '4Ato4B_1', 'Sto4B_0', 'Sto4B_1',\
                        ]
    DISTANCE_OF_ROUTE = {"route1A5A": 830, "route1B5B": 830, "route1NS": 490, "route1SN": 490, "route2NS": 490, "route2SN": 490, \
                         "route3NS": 490, "route3SN": 490, "route4NS": 490, "route4SN": 490}
    TRAFFIC_SIGNAL_LIGHT_NAMES = ['node1', 'node2', 'node3', 'node4', 'node1B', 'node2B', 'node3B', 'node4B']


if args.heavy_traffic:
    sumoCmd.extend(['-r', './network/%s.heavy.route.xml' % args.net_file])
    LOG_QUEUE_LENGTH_FILE_NAME = './log/%s/heavy-traffic/MaxPressure/queue-length.txt' % args.net_file
    LOG_VEHICLE_FILE_NAME = './log/%s/heavy-traffic/MaxPressure/vehicle.txt' % args.net_file
    LOG_TRAFFIC_LIGHT_FILE_NAME = './log/%s/heavy-traffic/MaxPressure/traffic-light.txt' % args.net_file
elif args.light_traffic:
    sumoCmd.extend(['-r', './network/%s.light.route.xml' % args.net_file])
    LOG_QUEUE_LENGTH_FILE_NAME = './log/%s/light-traffic/MaxPressure/queue-length.txt' % args.net_file
    LOG_VEHICLE_FILE_NAME = './log/%s/light-traffic/MaxPressure/vehicle.txt' % args.net_file
    LOG_TRAFFIC_LIGHT_FILE_NAME = './log/%s/light-traffic/MaxPressure/traffic-light.txt' % args.net_file

MIN_GREEN_VEHICLE = 20
MAX_RED_VEHICLE = 30

class Simulation_SOTL():
    def __init__(self):
        traci.start(sumoCmd)
        self.queue_length_per_step = []

        os.makedirs(os.path.dirname(LOG_QUEUE_LENGTH_FILE_NAME), exist_ok=True)
        self.log_QL_file = open(LOG_QUEUE_LENGTH_FILE_NAME, "w")
        self.log_QL_file.write('step')
        [self.log_QL_file.write(',%s' % lane) for lane in LIST_INCOMING_LANES]
        self.log_QL_file.write('\n')

        os.makedirs(os.path.dirname(LOG_TRAFFIC_LIGHT_FILE_NAME), exist_ok=True)
        self.log_traffic_light = open(LOG_TRAFFIC_LIGHT_FILE_NAME, "w")
        self.log_traffic_light.write('step')
        [self.log_traffic_light.write(',%s' % tls) for tls in TRAFFIC_SIGNAL_LIGHT_NAMES]
        self.log_traffic_light.write('\n')
        
        os.makedirs(os.path.dirname(LOG_VEHICLE_FILE_NAME), exist_ok=True)
        self.log_Veh_file = open(LOG_VEHICLE_FILE_NAME, "w")
        self.log_Veh_file.write('VehID,DepartedTime,ArrivedTime,RouteID,TravelTime,AverageSpeed\n')            
        self.vehicle_tracker = dict()



    def nextStep(self):
        self.log_step()
        traci.simulationStep()

    def log_step(self, exceedTimeSimulation=False):
        # QUEUE_LENGTH
        q_length = []
        for lane in LIST_INCOMING_LANES_LOG_QUEUE_LENGTH:
            q_length.append(traci.lane.getLastStepHaltingNumber(lane))
        str_q_length = str(traci.simulation.getTime())
        for q in q_length:
            str_q_length += ',%d' % q
        self.log_QL_file.write(str_q_length + '\n')
        self.queue_length_per_step.append(q_length)

        # LOG VEHICLE INFORMATION
        step = traci.simulation.getTime()
        for vehicleId in traci.simulation.getDepartedIDList():
            self.vehicle_tracker[vehicleId] = dict()
            self.vehicle_tracker[vehicleId]['departed'] = step
            self.vehicle_tracker[vehicleId]['routeID'] = traci.vehicle.getRouteID(vehicleId)
            self.vehicle_tracker[vehicleId]['arrived'] = None
        for vehicleId in traci.simulation.getArrivedIDList():
            self.vehicle_tracker[vehicleId]['arrived'] = step
            self.vehicle_tracker[vehicleId]['travel_time'] = self.vehicle_tracker[vehicleId]['arrived'] - self.vehicle_tracker[vehicleId]['departed']
            self.vehicle_tracker[vehicleId]['average_speed'] = DISTANCE_OF_ROUTE[self.vehicle_tracker[vehicleId]['routeID']]/self.vehicle_tracker[vehicleId]['travel_time']
            self.log_Veh_file.write('%s,%f,%f,%s,%f,%f\n' % (vehicleId,   self.vehicle_tracker[vehicleId]['departed'],\
                                                                        self.vehicle_tracker[vehicleId]['arrived'],\
                                                                        self.vehicle_tracker[vehicleId]['routeID'],\
                                                                        self.vehicle_tracker[vehicleId]['travel_time'],\
                                                                        self.vehicle_tracker[vehicleId]['average_speed']))
        if exceedTimeSimulation:
            for vehicleId in self.vehicle_tracker.keys():
                if self.vehicle_tracker[vehicleId]['arrived'] == None:
                    self.vehicle_tracker[vehicleId]['arrived'] = step
                    self.vehicle_tracker[vehicleId]['travel_time'] = self.vehicle_tracker[vehicleId]['arrived'] - self.vehicle_tracker[vehicleId]['departed']
                    self.vehicle_tracker[vehicleId]['average_speed'] = DISTANCE_OF_ROUTE[self.vehicle_tracker[vehicleId]['routeID']]/self.vehicle_tracker[vehicleId]['travel_time']
                    self.log_Veh_file.write('%s,%f,%f,%s,%f,%f\n' % (vehicleId,   self.vehicle_tracker[vehicleId]['departed'],\
                                                                                self.vehicle_tracker[vehicleId]['arrived'],\
                                                                                self.vehicle_tracker[vehicleId]['routeID'],\
                                                                                self.vehicle_tracker[vehicleId]['travel_time'],\
                                                                                self.vehicle_tracker[vehicleId]['average_speed']))
        # LOG TRAFFIC LIGHT
        s = str(step)
        for tls in TRAFFIC_SIGNAL_LIGHT_NAMES:
            s += ',%d' % traci.trafficlight.getPhase(tls)
        self.log_traffic_light.write(s + '\n')

    def changePhase(self):
        flags = dict()
        allowed_pressure = 0
        disallowed_pressure = 0
        for tls in TRAFFIC_SIGNAL_LIGHT_NAMES:
            currentPhase = traci.trafficlight.getPhase(tls)

            pressure_array = []
            # 8 is numbers of lane
            for idx in range(8):
                pressure_array.append(traci.lane.getLastStepVehicleNumber(LIST_INCOMING_LANES[tls][idx]) \
                                        - traci.lane.getLastStepVehicleNumber(LIST_OUTGOING_LANES[tls][idx]))   
            willChange = False
            if (currentPhase == 0 and np.sum(pressure_array[0:4]) < np.sum(pressure_array[4:8]))\
                    or (currentPhase == 2 and np.sum(pressure_array[0:4]) > np.sum(pressure_array[4:8])):
                traci.trafficlight.setPhase(tls, currentPhase + 1)
                flags[tls] = True
        if len(flags.keys()) <= 0:
            return
        for _ in range(3):
            self.nextStep()
        for tls in flags.keys():
            currentPhase = traci.trafficlight.getPhase(tls)
            if currentPhase == 3:
                traci.trafficlight.setPhase(tls, 0)
            else:
                traci.trafficlight.setPhase(tls, currentPhase + 1)

    def isFinised(self):
        if traci.simulation.getMinExpectedNumber() <= 0:
            self.log_step()
            traci.close()
            
            t = PrettyTable(['Feature', 'Value'])
            t.add_row(['Average Queue Length', np.mean(self.queue_length_per_step)])
            t.add_row(['Average Travel Time', np.mean([veh['travel_time'] for _, veh in self.vehicle_tracker.items()])])
            t.add_row(['Average Speed', np.mean([veh['average_speed'] for _, veh in self.vehicle_tracker.items()])])  
            print(t)

            return True
        elif traci.simulation.getTime() >= args.max_step:
            self.log_step(exceedTimeSimulation=True)
            traci.close()

            t = PrettyTable(['Feature', 'Value'])
            t.add_row(['Average Queue Length', np.mean(self.queue_length_per_step)])
            t.add_row(['Average Travel Time', np.mean([veh['travel_time'] for _, veh in self.vehicle_tracker.items()])])
            t.add_row(['Average Speed', np.mean([veh['average_speed'] for _, veh in self.vehicle_tracker.items()])])  
            print(t)
            return True
            
        return False

    def makeAction(self):
        self.changePhase()
        self.nextStep()

s = Simulation_SOTL()
while s.isFinised() != True:
    s.makeAction()
