import os
import sys
import argparse
import numpy as np
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci


parser = argparse.ArgumentParser() 
parser.add_argument("--gui", action='store_true', help = "Enable gui")
parser.add_argument('--net-file', type=str, dest="net_file", help='Name of the net file')
parser.add_argument('--route-file', type=str, dest="route_file", help='Name of the route file')
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
    
if args.heavy_traffic:
    sumoCmd.extend(['-r', './network/%s.heavy.route.xml' % args.route_file])
    LOG_FILE_NAME = './log/FT-queue-length-%s-heavy-traffic.txt' % args.net_file
elif args.light_traffic:
    sumoCmd.extend(['-r', './network/%s.light.route.xml' % args.route_file])
    LOG_FILE_NAME = './log/FT-queue-length-%s-light-traffic.txt' % args.net_file

TRAFFIC_SIGNAL_LIGHT_NAMES = ['node1', 'node2', 'node3', 'node4']
MOVEMENT_DISTANCE = [150*5 + 4*20, 150*2 + 20]



class Simulation_FT():
    def __init__(self):
        traci.start(sumoCmd)
        self.current_phase_duration = 0
        self.queue_length_per_step = []
        self.log_file = open(LOG_FILE_NAME, "w")
        self.log_file.write('STEP,W11,W12,N11,N12,E11,E12,S11,S12,W21,W22,N21,N22,E21,E22,S21,S22,W31,W32,N31,N32,E31,E32,S31,S32,W41,W42,N41,N42,E41,E42,S41,S42\n')

    def nextStep(self):
        q_length = []
        for lane in LIST_INCOMING_LANES:
            q_length.append(traci.lane.getLastStepHaltingNumber(lane))
        str_q_length = str(traci.simulation.getTime())
        for q in q_length:
            str_q_length += ',%d' % q
        self.log_file.write(str_q_length + '\n')
        self.queue_length_per_step.append(q_length)

        self.current_phase_duration += 1
        traci.simulationStep()

    def changePhase(self):
        self.current_phase_duration = 0
        currentPhase = traci.trafficlight.getPhase(TRAFFIC_SIGNAL_LIGHT_NAMES[0])
        if currentPhase == 0:
            for name in TRAFFIC_SIGNAL_LIGHT_NAMES:
                traci.trafficlight.setPhase(name, 1)
            for _ in range(3):
                self.nextStep()
            for name in TRAFFIC_SIGNAL_LIGHT_NAMES:
                traci.trafficlight.setPhase(name, 2)
        elif currentPhase == 2:
            for name in TRAFFIC_SIGNAL_LIGHT_NAMES:
                traci.trafficlight.setPhase(name, 3)
            for _ in range(3):
                self.nextStep()
            for name in TRAFFIC_SIGNAL_LIGHT_NAMES:
                traci.trafficlight.setPhase(name, 0)
        else:
            print("error in change phase")

    def isFinised(self):
        if traci.simulation.getMinExpectedNumber() <= 0:
            traci.close()
            print(np.mean(self.queue_length_per_step))
            return True
        return False

    def makeAction(self):
        if self.current_phase_duration >= 15:
            self.changePhase()
        else:
            self.nextStep()

s = Simulation_FT()
while s.isFinised() != True:
    s.makeAction()
