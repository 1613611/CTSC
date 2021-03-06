import os
import sys
import numpy as np
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci

LIST_INCOMING_LANES = {}
STEP_SIZE = 5
DISTANCE_OF_ROUTE = {}
LIST_INCOMING_LANES_LOG_QUEUE_LENGTH = {}
INTERSECTION_NAMES = []
LIST_OUTGOING_LANES = {}
class Environment():
    def __init__(self, args):
        global LIST_INCOMING_LANES, STEP_SIZE, DISTANCE_OF_ROUTE, LIST_INCOMING_LANES_LOG_QUEUE_LENGTH, INTERSECTION_NAMES, LIST_OUTGOING_LANES
        if args.gui:
            sumoBinary = "/usr/bin/sumo-gui"
        else:
            sumoBinary = "/usr/bin/sumo"
        sumoConfig = ["-c", "./network/network.sumocfg"]
        sumoCmd = [sumoBinary, sumoConfig[0], sumoConfig[1]]
        sumoCmd.extend(['-n', './network/%s.net.xml' % args.net_file])
        if args.heavy_traffic:
            sumoCmd.extend(['-r', './network/%s.heavy.route.xml' % args.net_file])
        elif args.light_traffic:
            sumoCmd.extend(['-r', './network/%s.light.route.xml' % args.net_file])
        self.sumoCmd = sumoCmd.copy()

        if args.net_file == '4x1-two-way':
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

            DISTANCE_OF_ROUTE = {"route1": 830, "route2": 830, "route1A": 320, "route1B": 320, "route2A": 320, "route2B": 320, \
                                "route3A": 320, "route3B": 320, "route4A": 320, "route4B": 320}    
            LIST_INCOMING_LANES_LOG_QUEUE_LENGTH = [ '0to1_0', 'NtoC_1_0', '2to1_0', 'StoC_1_0', '1to2_0', 'NtoC_2_0', '3to2_0', 'StoC_2_0',\
                        '2to3_0', 'NtoC_3_0', '4to3_0', 'StoC_3_0', '3to4_0', 'NtoC_4_0', '5to4_0', 'StoC_4_0',\
                        '0to1_1', 'NtoC_1_1', '2to1_1', 'StoC_1_1', '1to2_1', 'NtoC_2_1', '3to2_1', 'StoC_2_1',\
                        '2to3_1', 'NtoC_3_1', '4to3_1', 'StoC_3_1', '3to4_1', 'NtoC_4_1', '5to4_1', 'StoC_4_1']
            INTERSECTION_NAMES = ['node1', 'node2', 'node3', 'node4']

        elif args.net_file == '4x1-one-way':
            LIST_INCOMING_LANES =   {
                            'node1': ['0to1_0', '0to1_1', '0to1_2', '0to1_3', 'NtoC_1_0', 'NtoC_1_1', 'StoC_1_0', 'StoC_1_1'],
                            'node2': ['1to2_0', '1to2_1', '1to2_2', '1to2_3', 'NtoC_2_0', 'NtoC_2_1', 'StoC_2_0', 'StoC_2_1'],
                            'node3': ['2to3_0', '2to3_1', '2to3_2', '2to3_3', 'NtoC_3_0', 'NtoC_3_1', 'StoC_3_0', 'StoC_3_1'],
                            'node4': ['3to4_0', '3to4_1', '3to4_2', '3to4_3', 'NtoC_4_0', 'NtoC_4_1', 'StoC_4_0', 'StoC_4_1']
                        }
            LIST_OUTGOING_LANES =   {
                                    'node1': ['1to2_0', '1to2_1', '1to2_2', '1to2_3', 'CtoS_1_0', 'CtoS_1_1', 'CtoN_1_0', 'CtoN_1_1'],
                                    'node2': ['2to3_0', '2to3_1', '2to3_2', '2to3_3', 'CtoS_2_0', 'CtoS_2_1', 'CtoN_2_0', 'CtoN_2_1'],
                                    'node3': ['3to4_0', '3to4_1', '3to4_2', '3to4_3', 'CtoS_3_0', 'CtoS_3_1', 'CtoN_3_0', 'CtoN_3_1'],
                                    'node4': ['4to5_0', '4to5_1', '4to5_2', '4to5_3', 'CtoS_4_0', 'CtoS_4_1', 'CtoN_4_0', 'CtoN_4_1'],
                                }

            LIST_INCOMING_LANES_LOG_QUEUE_LENGTH = ['0to1_0', '0to1_1', '0to1_2', '0to1_3', 'NtoC_1_0', 'NtoC_1_1', 'StoC_1_0', 'StoC_1_1', \
                                                '1to2_0', '1to2_1', '1to2_2', '1to2_3', 'NtoC_2_0', 'NtoC_2_1', 'StoC_2_0', 'StoC_2_1', \
                                                '2to3_0', '2to3_1', '2to3_2', '2to3_3', 'NtoC_3_0', 'NtoC_3_1', 'StoC_3_0', 'StoC_3_1', \
                                                '3to4_0', '3to4_1', '3to4_2', '3to4_3', 'NtoC_4_0', 'NtoC_4_1', 'StoC_4_0', 'StoC_4_1']
            DISTANCE_OF_ROUTE = {"route1": 830, "route1A": 320, "route1B": 320, "route2A": 320, "route2B": 320, \
                                "route3A": 320, "route3B": 320, "route4A": 320, "route4B": 320}    
            INTERSECTION_NAMES = ['node1', 'node2', 'node3', 'node4']


        elif args.net_file == '4x2-intersections':
            LIST_INCOMING_LANES =   {
                                        'node1': ['0Ato1A_0', '0Ato1A_1', '0Ato1A_2', '0Ato1A_3', 'Nto1A_0', 'Nto1A_1', '1Bto1A_0', '1Bto1A_1'],
                                        'node2': ['1Ato2A_0', '1Ato2A_1', '1Ato2A_2', '1Ato2A_3', 'Nto2A_0', 'Nto2A_1', '2Bto2A_0', '2Bto2A_1'],
                                        'node3': ['2Ato3A_0', '2Ato3A_1', '2Ato3A_2', '2Ato3A_3', 'Nto3A_0', 'Nto3A_1', '3Bto3A_0', '3Bto3A_1'],
                                        'node4': ['3Ato4A_0', '3Ato4A_1', '3Ato4A_2', '3Ato4A_3', 'Nto4A_0', 'Nto4A_1', '4Bto4A_0', '4Bto4A_1'],
                                        'node1B': ['2Bto1B_0', '2Bto1B_1', '2Bto1B_2', '2Bto1B_3', '1Ato1B_0', '1Ato1B_1', 'Sto1B_0', 'Sto1B_1'],
                                        'node2B': ['3Bto2B_0', '3Bto2B_1', '3Bto2B_2', '3Bto2B_3', '2Ato2B_0', '2Ato2B_1', 'Sto2B_0', 'Sto2B_1'],
                                        'node3B': ['4Bto3B_0', '4Bto3B_1', '4Bto3B_2', '4Bto3B_3', '3Ato3B_0', '3Ato3B_1', 'Sto3B_0', 'Sto3B_1'],
                                        'node4B': ['5Bto4B_0', '5Bto4B_1', '5Bto4B_2', '5Bto4B_3', '4Ato4B_0', '4Ato4B_1', 'Sto4B_0', 'Sto4B_1']
                                    }
            LIST_OUTGOING_LANES =   {
                                    'node1': ['1Ato2A_0', '1Ato2A_1', '1Ato2A_2', '1Ato2A_3', '1Ato1B_0', '1Ato1B_1', '1AtoN_0', '1AtoN_1'],
                                    'node2': ['2Ato3A_0', '2Ato3A_1', '2Ato3A_2', '2Ato3A_3', '2Ato2B_0', '2Ato2B_1', '2AtoN_0', '2AtoN_1'],
                                    'node3': ['3Ato4A_0', '3Ato4A_1', '3Ato4A_2', '3Ato4A_3', '3Ato3B_0', '3Ato3B_1', '3AtoN_0', '3AtoN_1'],
                                    'node4': ['4Ato5A_0', '4Ato5A_1', '4Ato5A_2', '4Ato5A_3', '4Ato4B_0', '3Ato3B_1', '4AtoN_0', '4AtoN_1'],
                                    'node1B': ['1Bto0B_0', '1Bto0B_1', '1Bto0B_2', '1Bto0B_3', '1BtoS_0', '1BtoS_1', '1Bto1A_0', '1Bto1A_1'],                            
                                    'node2B': ['2Bto1B_0', '2Bto1B_1', '2Bto1B_2', '2Bto1B_3', '2BtoS_0', '2BtoS_1', '2Bto2A_0', '2Bto2A_1'],                             
                                    'node3B': ['3Bto2B_0', '3Bto2B_1', '3Bto2B_2', '3Bto2B_3', '3BtoS_0', '3BtoS_1', '3Bto3A_0', '3Bto3A_1'],
                                    'node4B': ['4Bto3B_0', '4Bto3B_1', '4Bto3B_2', '4Bto3B_3', '4BtoS_0', '4BtoS_1', '4Bto4A_0', '4Bto4A_1'],                                                                                                               
                                }
            LIST_INCOMING_LANES_LOG_QUEUE_LENGTH = [ '0Ato1A_0', '0Ato1A_1', '0Ato1A_2', '0Ato1A_3', 'Nto1A_0', 'Nto1A_1', '1Bto1A_0', '1Bto1A_1',\
                                    '1Ato2A_0', '1Ato2A_1', '1Ato2A_2', '1Ato2A_3', 'Nto2A_0', 'Nto2A_1', '2Bto2A_0', '2Bto2A_1',\
                                    '2Ato3A_0', '2Ato3A_1', '2Ato3A_2', '2Ato3A_3', 'Nto3A_0', 'Nto3A_1', '3Bto3A_0', '3Bto3A_1',\
                                    '3Ato4A_0', '3Ato4A_1', '3Ato4A_2', '3Ato4A_3', 'Nto4A_0', 'Nto4A_1', '4Bto4A_0', '4Bto4A_1',\
                                    '2Bto1B_0', '2Bto1B_1', '2Bto1B_2', '2Bto1B_3', '1Ato1B_0', '1Ato1B_1', 'Sto1B_0', 'Sto1B_1',\
                                    '3Bto2B_0', '3Bto2B_1', '3Bto2B_2', '3Bto2B_3', '2Ato2B_0', '2Ato2B_1', 'Sto2B_0', 'Sto2B_1',\
                                    '4Bto3B_0', '4Bto3B_1', '4Bto3B_2', '4Bto3B_3', '3Ato3B_0', '3Ato3B_1', 'Sto3B_0', 'Sto3B_1',\
                                    '5Bto4B_0', '5Bto4B_1', '5Bto4B_2', '5Bto4B_3', '4Ato4B_0', '4Ato4B_1', 'Sto4B_0', 'Sto4B_1',\
                                ]
            DISTANCE_OF_ROUTE = {"route1A5A": 830, "route5B1B": 830, "route1NS": 490, "route1SN": 490, "route2NS": 490, "route2SN": 490, \
                                "route3NS": 490, "route3SN": 490, "route4NS": 490, "route4SN": 490}
            INTERSECTION_NAMES = ['node1', 'node2', 'node3', 'node4', 'node1B', 'node2B', 'node3B', 'node4B']


        STEP_SIZE = args.step_size
        self.net_file = args.net_file
        self.max_step = args.max_step

    def reset(self):
        self.queue_length_per_step = []
        self.vehicle_tracker = dict()
        self.traffic_light_log = []
        traci.start(self.sumoCmd)

    def get_log(self):
        return self.queue_length_per_step, self.vehicle_tracker, self.traffic_light_log

    def get_number_vehicle_on_incoming_lane(self, lane, trust_region_length=50, lane_length=150):
        veh_list = traci.lane.getLastStepVehicleIDs(lane)
        n = 0
        for veh in veh_list:
            if (lane_length - traci.vehicle.getLanePosition(veh) < trust_region_length):
                n += 1
        return n
    
    def get_number_vehicle_on_outgoing_lane(self, lane, trust_region_length=50, lane_length=150):
        veh_list = traci.lane.getLastStepVehicleIDs(lane)
        n = 0
        for veh in veh_list:
            if (traci.vehicle.getLanePosition(veh) < trust_region_length):
                n += 1
        return n


    def get_observation(self, agent):
        currentPhase = traci.trafficlight.getPhase(agent)
        number_waiting_veh_on_coming_lanes_allowed = 0
        number_veh_on_coming_lanes_allowed = 0
        number_waiting_veh_on_coming_lanes_disallowed = 0
        number_veh_on_coming_lanes_disallowed = 0
        if currentPhase == 2:
            for lane in LIST_INCOMING_LANES[agent][0:4]:
                number_waiting_veh_on_coming_lanes_allowed += traci.lane.getLastStepHaltingNumber(lane)
                number_veh_on_coming_lanes_allowed += self.get_number_vehicle_on_incoming_lane(lane)
            for lane in LIST_INCOMING_LANES[agent][4:]:
                number_waiting_veh_on_coming_lanes_disallowed += traci.lane.getLastStepHaltingNumber(lane)
                number_veh_on_coming_lanes_disallowed += self.get_number_vehicle_on_incoming_lane(lane)
        elif currentPhase == 0:
            for lane in LIST_INCOMING_LANES[agent][0:4]:
                number_waiting_veh_on_coming_lanes_disallowed += traci.lane.getLastStepHaltingNumber(lane)
                number_veh_on_coming_lanes_disallowed += self.get_number_vehicle_on_incoming_lane(lane)
            for lane in LIST_INCOMING_LANES[agent][4:]:
                number_waiting_veh_on_coming_lanes_allowed += traci.lane.getLastStepHaltingNumber(lane)
                number_veh_on_coming_lanes_allowed += self.get_number_vehicle_on_incoming_lane(lane)
        else:
            print('error in get_observation')
        return [number_waiting_veh_on_coming_lanes_allowed, number_veh_on_coming_lanes_allowed, \
                number_waiting_veh_on_coming_lanes_disallowed, number_veh_on_coming_lanes_disallowed]
    
    def get_reward(self, agent):
        reward = 0
        for lane in LIST_INCOMING_LANES[agent]:
            reward -= traci.lane.getLastStepHaltingNumber(lane)
        for lane in LIST_OUTGOING_LANES[agent]:
            reward += self.get_number_vehicle_on_outgoing_lane(lane)

        return reward
        
    def set_action(self, actions):
        flags = dict()
        for agent, action in actions.items():
            if action == 1:
                flags[agent] = True
                self.change_phase(agent)
        for _ in range(3):
            self.next_step()
        for agent, flag in flags.items():
            if flag:
                self.change_phase(agent)
        for _ in range(STEP_SIZE - 3):
            self.next_step()
        
        rewards = dict()
        next_states = dict()
        for agent,_ in actions.items():
            rewards[agent] = self.get_reward(agent)
            next_states[agent] = self.get_observation(agent)
        return rewards, next_states, self.isFinish()

    def next_step(self):
        self.log_step()
        traci.simulationStep()

    def log_step(self):
        q_length = []
        for lane in LIST_INCOMING_LANES_LOG_QUEUE_LENGTH:
            q_length.append(traci.lane.getLastStepHaltingNumber(lane))
        self.queue_length_per_step.append(q_length)

        step = traci.simulation.getTime()
        for vehicleId in traci.simulation.getDepartedIDList():
            self.vehicle_tracker[vehicleId] = dict()
            self.vehicle_tracker[vehicleId]['departed'] = step
            self.vehicle_tracker[vehicleId]['routeID'] = traci.vehicle.getRouteID(vehicleId)
        for vehicleId in traci.simulation.getArrivedIDList():
            self.vehicle_tracker[vehicleId]['arrived'] = step
            self.vehicle_tracker[vehicleId]['travel_time'] = self.vehicle_tracker[vehicleId]['arrived'] - self.vehicle_tracker[vehicleId]['departed']
            self.vehicle_tracker[vehicleId]['average_speed'] = DISTANCE_OF_ROUTE[self.vehicle_tracker[vehicleId]['routeID']]/self.vehicle_tracker[vehicleId]['travel_time']

        tls_log = []
        for tls in INTERSECTION_NAMES:
            tls_log.append(traci.trafficlight.getPhase(tls))
        self.traffic_light_log.append(tls_log)


    def change_phase(self, intersection):
        cur_phase = traci.trafficlight.getPhase(intersection)
        if  cur_phase == 3:
            traci.trafficlight.setPhase(intersection, 0)
        else:           
            traci.trafficlight.setPhase(intersection, cur_phase + 1)

    def finish_by_max_step(self):
        step = traci.simulation.getTime()
        if step >= self.max_step:
            for vehicleId in self.vehicle_tracker.keys():
                if 'arrived' not in self.vehicle_tracker[vehicleId].keys():
                    self.vehicle_tracker[vehicleId]['arrived'] = step
                    self.vehicle_tracker[vehicleId]['travel_time'] = self.vehicle_tracker[vehicleId]['arrived'] - self.vehicle_tracker[vehicleId]['departed']
                    self.vehicle_tracker[vehicleId]['average_speed'] = DISTANCE_OF_ROUTE[self.vehicle_tracker[vehicleId]['routeID']]/self.vehicle_tracker[vehicleId]['travel_time']
            return True
        return False
        
    def isFinish(self):
        if traci.simulation.getMinExpectedNumber() <= 0 and traci.simulation.getTime() > 100:
            self.log_step()
            traci.close()
            return True
        if self.finish_by_max_step():
            traci.close()
            return True
        return False