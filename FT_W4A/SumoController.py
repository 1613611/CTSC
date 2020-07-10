import socket
import queue 
import threading 
import time
import json
from _thread import *

import os, sys
import math
import numpy as np
import random
import string

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci

sumoBinary = "/usr/bin/sumo"
sumoConfig = ["-c", "./one_run/cross.sumocfg"]

sumoCmd = [sumoBinary, sumoConfig[0], sumoConfig[1]]

TRAFFIC_SIGNAL_NAME = 'node0'

N_STEP = 72000

LIST_LANE_IN = ['WtoC_0', 'WtoC_1', 'WtoC_2', 'EtoC_0', 'EtoC_1', 'EtoC_2', 'StoC_0', 'StoC_1', 'StoC_2', 'NtoC_0', 'NtoC_1', 'NtoC_2']
LIST_LANE_OUT = ['CtoW_0', 'CtoW_1', 'CtoW_2', 'CtoE_0', 'CtoE_1', 'CtoE_2', 'CtoS_0', 'CtoS_1', 'CtoS_2', 'CtoN_0', 'CtoN_1', 'CtoN_2']

LIST_WAIT_LANE_IN_0 = ['StoC_0', 'StoC_1', 'StoC_2', 'NtoC_0', 'NtoC_1', 'NtoC_2']
LIST_WAIT_LANE_IN_1 = ['WtoC_0', 'WtoC_1', 'WtoC_2', 'EtoC_0', 'EtoC_1', 'EtoC_2']

LIST_DETECTOR_W_E = ['W_0', 'W_1', 'W_2', 'E_0', 'E_1', 'E_2']
LIST_DETECTOR_N_S = ['S_0', 'S_1', 'S_2', 'N_0', 'N_1', 'N_2']

LIST_LANE_IN_N_S = ['StoC_0', 'StoC_1', 'StoC_2', 'NtoC_0', 'NtoC_1', 'NtoC_2']
LIST_LANE_IN_W_E = ['WtoC_0', 'WtoC_1', 'WtoC_2', 'EtoC_0', 'EtoC_1', 'EtoC_2']
LIST_LANE_OUT_N_S = ['CtoS_0', 'CtoS_1', 'CtoS_2', 'CtoN_0', 'CtoN_1', 'CtoN_2']
LIST_LANE_OUT_W_E = ['CtoW_0', 'CtoW_1', 'CtoW_2', 'CtoE_0', 'CtoE_1', 'CtoE_2']


def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

STEP_SIZE = 1

class SumoController():
    def __init__(self):
        self.step = 0
        self.average_waiting_time = 0
        self.last_average_waiting_time = 0
        self.total_reward = 0
        label = randomword(64)
        traci.start(sumoCmd, label = label)
        traci.switch(label)
        traci.vehicletype.setMaxSpeed('moped', 34.61)
        traci.vehicletype.setMaxSpeed('passenger', 178.37)
        traci.vehicletype.setImperfection('moped', 0.36)
        traci.vehicletype.setImperfection('passenger', 0.33)
        traci.vehicletype.setSpeedDeviation('moped', 0.11)
        traci.vehicletype.setSpeedDeviation('passenger', 0.06)
        traci.vehicletype.setMinGapLat('moped', 0.57)
        traci.vehicletype.setMinGapLat('passenger', 0.88)
        traci.vehicletype.setAccel('moped', 1.08)
        traci.vehicletype.setAccel('passenger', 3.51)
        traci.vehicletype.setDecel('moped', 3.51)
        traci.vehicletype.setDecel('passenger', 10.95)


        self.N_change_actions = 0
        self.average_travel_time = 0
        self.generate_object = dict()
        self.Nlen = 0
        self.N_vehicle_finished = 0

        self.tracking_vehicle = dict()
        self.n_tracking = 0
        self.duration = 0.0
        self.log_reward = dict()
        self.log_reward['q_length'] = []
        self.log_reward['delay'] = []
        self.log_reward['duration'] = []
        self.phase_duration = 0


    def nextStep(self):
        if self.step % 100 == 0:
            print(self.step)
        self.phase_duration += 1

        q_length = 0
        delay = 0
        for lane in LIST_LANE_IN:
            q_length += traci.lane.getLastStepHaltingNumber(lane)
            delay += 1 - traci.lane.getLastStepMeanSpeed(lane)/traci.lane.getMaxSpeed(lane)
        self.log_reward['q_length'].append(q_length)
        self.log_reward['delay'].append(delay)
        
        for vehicleId in traci.simulation.getDepartedIDList():
            self.generate_object[vehicleId] = dict()
            self.generate_object[vehicleId]['departed'] = self.step
            self.generate_object[vehicleId]['lane'] = traci.vehicle.getLaneID(vehicleId)
            pos = traci.vehicle.getPosition(vehicleId)
            self.generate_object[vehicleId]['position'] = [pos for _ in range(5)]
            self.Nlen += 1
            self.tracking_vehicle[vehicleId] = dict()
            self.tracking_vehicle[vehicleId]['departed'] = self.step
            self.tracking_vehicle[vehicleId]['lane'] = traci.vehicle.getLaneID(vehicleId)
        for vehicleId in traci.simulation.getArrivedIDList():
            self.average_travel_time += self.step - self.generate_object[vehicleId]['departed']
            del self.generate_object[vehicleId]
            self.N_vehicle_finished += 1
        for vehicleId in list(self.tracking_vehicle):
            if traci.vehicle.getLaneID(vehicleId) not in LIST_LANE_IN:
                self.duration += self.step - self.tracking_vehicle[vehicleId]['departed'] - 1
                del self.tracking_vehicle[vehicleId]
                self.n_tracking += 1
            # self.generate_object[vehicleId]['position'].pop(0)
            # self.generate_object[vehicleId]['position'].append(traci.vehicle.getPosition(vehicleId))

        vehicle_id_entering = []
        for lane in LIST_LANE_IN:
            vehicle_id_entering.extend(traci.lane.getLastStepVehicleIDs(lane))
        travel_time_duration = 0
        for vehicle_id in vehicle_id_entering:
            travel_time_duration += (traci.simulation.getCurrentTime() / 1000 - self.generate_object[vehicle_id]['departed'])
        self.log_reward['duration'].append(travel_time_duration)

        traci.simulationStep()
        self.step += 1

    def changePhase(self):
        self.phase_duration = 0
        self.N_change_actions += 1
        currentPhase = traci.trafficlight.getPhase(TRAFFIC_SIGNAL_NAME)
        if currentPhase == 0:
            traci.trafficlight.setPhase(TRAFFIC_SIGNAL_NAME, 1)
            for _ in range(3):
                self.nextStep()
            traci.trafficlight.setPhase(TRAFFIC_SIGNAL_NAME, 2)
        elif currentPhase == 2:
            traci.trafficlight.setPhase(TRAFFIC_SIGNAL_NAME, 3)
            for _ in range(3):
                self.nextStep()
            traci.trafficlight.setPhase(TRAFFIC_SIGNAL_NAME, 0)
        else:
            print("errrororroror")
    def isFinised(self):
        if traci.simulation.getMinExpectedNumber() <= 0:
            for vehicleId in self.generate_object:
                self.average_travel_time += self.step - self.generate_object[vehicleId]['departed']
            traci.close()
            return True
        return False

    def getTotalReward(self):
        return self.average_travel_time * 1.0 / self.Nlen

    def getNumberOfRequest(self, isWE):
        tmp = 0
        if isWE:
            for detector in LIST_DETECTOR_W_E:
                tmp += traci.inductionloop.getLastStepVehicleNumber(detector)
        else:
            for detector in LIST_DETECTOR_N_S:
                tmp += traci.inductionloop.getLastStepVehicleNumber(detector)
        return tmp

    def getNumberOfApproaching(self, isWE):
        tmp = 0
        if isWE:
            for lane in LIST_LANE_IN_W_E:
                tmp += traci.lane.getLastStepVehicleNumber(lane)
        else:
            for lane in LIST_LANE_IN_N_S:
                tmp += traci.lane.getLastStepVehicleNumber(lane)
        return tmp

    def makeAction(self, action):
        currentPhase = traci.trafficlight.getPhase(TRAFFIC_SIGNAL_NAME)
        # if (currentPhase == 0 and self.getNumberOfApproaching(False) > 1 and self.getNumberOfApproaching(True) <= 0) \
        #     or (currentPhase == 2 and self.getNumberOfApproaching(True) > 1 and self.getNumberOfApproaching(False) <= 0): 
            # \            or self.phase_duration >= 100:
        if self.phase_duration >= 15:
            self.changePhase()
            for _ in range(STEP_SIZE):
                self.nextStep()
        else:
            # for _ in range(STEP_SIZE):
            self.nextStep()
        
        return 0
        

    def getState(self):
        return 0, np.asarray([])
        
    def close(self):
        traci.close()

    def getDuration(self):
        return  self.duration/self.n_tracking

    def getLogReward(self):
        return self.log_reward
