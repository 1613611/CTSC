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
args = parser.parse_args()

if args.gui:
    sumoBinary = "/usr/bin/sumo-gui"
else:
    sumoBinary = "/usr/bin/sumo"
sumoConfig = ["-c", "./network/network.sumocfg"]
sumoCmd = [sumoBinary, sumoConfig[0], sumoConfig[1]]

sumoCmd.extend(['-n', './network/%s.net.xml' % args.net_file])

if args.net_file == '4-arterial-intersections':
    LIST_INCOMING_LANES = [ '0to1_0', 'NtoC_1_0', '2to1_0', 'StoC_1_0', '1to2_0', 'NtoC_2_0', '3to2_0', 'StoC_2_0',\
                            '2to3_0', 'NtoC_3_0', '4to3_0', 'StoC_3_0', '3to4_0', 'NtoC_4_0', '5to4_0', 'StoC_4_0',\
                            '0to1_1', 'NtoC_1_1', '2to1_1', 'StoC_1_1', '1to2_1', 'NtoC_2_1', '3to2_1', 'StoC_2_1',\
                            '2to3_1', 'NtoC_3_1', '4to3_1', 'StoC_3_1', '3to4_1', 'NtoC_4_1', '5to4_1', 'StoC_4_1']
    DISTANCE_OF_ROUTE = {"route1": 830, "route2": 830, "route1A": 320, "route1B": 320, "route2A": 320, "route2B": 320, \
                         "route3A": 320, "route3B": 320, "route4A": 320, "route4B": 320}    
    TRAFFIC_SIGNAL_LIGHT_NAMES = ['node1', 'node2', 'node3', 'node4']
    MOVEMENT_DISTANCE = [150*5 + 4*20, 150*2 + 20]

if args.heavy_traffic:
    sumoCmd.extend(['-r', './network/%s.heavy.route.xml' % args.net_file])
    LOG_QUEUE_LENGTH_FILE_NAME = './log/%s/heavy-traffic/FT/queue-length.txt' % args.net_file
    LOG_VEHICLE_FILE_NAME = './log/%s/heavy-traffic/FT/vehicle.txt' % args.net_file
    LOG_TRAFFIC_LIGHT_FILE_NAME = './log/%s/heavy-traffic/FT/traffic-light.txt' % args.net_file
elif args.light_traffic:
    sumoCmd.extend(['-r', './network/%s.light.route.xml' % args.net_file])
    LOG_QUEUE_LENGTH_FILE_NAME = './log/%s/light-traffic/FT/queue-length.txt' % args.net_file
    LOG_VEHICLE_FILE_NAME = './log/%s/light-traffic/FT/vehicle.txt' % args.net_file
    LOG_TRAFFIC_LIGHT_FILE_NAME = './log/%s/light-traffic/FT/traffic-light.txt' % args.net_file



PHASE_1_LENGTH = 25
PHASE_2_LENGTH = 5
OFFSET = 15
class Simulation_FT():
    def __init__(self):
        traci.start(sumoCmd)
        self.current_phase_duration = 0
        self.queue_length_per_step = []

        os.makedirs(os.path.dirname(LOG_TRAFFIC_LIGHT_FILE_NAME), exist_ok=True)
        self.log_traffic_light = open(LOG_TRAFFIC_LIGHT_FILE_NAME, "w")

        os.makedirs(os.path.dirname(LOG_QUEUE_LENGTH_FILE_NAME), exist_ok=True)
        self.log_QL_file = open(LOG_QUEUE_LENGTH_FILE_NAME, "w")
        if args.net_file == '4-arterial-intersections':
            self.log_QL_file.write('STEP,W11,W12,N11,N12,E11,E12,S11,S12,W21,W22,N21,N22,E21,E22,S21,S22,W31,W32,N31,N32,E31,E32,S31,S32,W41,W42,N41,N42,E41,E42,S41,S42\n')
            self.log_traffic_light.write('step,node1,node2,node3,node4\n')

        os.makedirs(os.path.dirname(LOG_VEHICLE_FILE_NAME), exist_ok=True)
        self.log_Veh_file = open(LOG_VEHICLE_FILE_NAME, "w")
        self.log_Veh_file.write('VehID,DepartedTime,ArrivedTime,RouteID,TravelTime,AverageSpeed\n')            
        self.vehicle_tracker = dict()


    def nextStep(self):
        self.log_step()
        self.current_phase_duration += 1
        traci.simulationStep()

    def log_step(self):
        # QUEUE_LENGTH
        q_length = []
        for lane in LIST_INCOMING_LANES:
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
        for vehicleId in traci.simulation.getArrivedIDList():
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

    def initPhase(self):
        for idx, name in enumerate(TRAFFIC_SIGNAL_LIGHT_NAMES):
            traci.trafficlight.setPhaseDuration(name, PHASE_1_LENGTH + idx*OFFSET)

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
        return False

    def makeAction(self):
        self.nextStep()

s = Simulation_FT()
s.initPhase()
while s.isFinised() != True:
    s.makeAction()
