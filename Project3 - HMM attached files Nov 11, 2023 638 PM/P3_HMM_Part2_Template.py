# Python program to HMM Part2 - template.
# Part2: figure out where the moving "hidden" car is.
# The car and your agent (car) live in a nxn periodic grid world.
# assume a shape of car is square, length is 1

import numpy as np
import pandas as pd
import sys
from scipy.stats import norm
import math


# Function: Get Belief
# ---------------------
# Updates beliefs based on recorded distance, transition prob, and your car's (agent's) pos.
# @input: gridSize, recording/observation, transition prob, car length = 1
# @return: your belief of the prob the "hidden" car is at each tile at each time step.
# Note: Your belief probabilities should sum to 1. (belief probabilities = posterior prob)

import numpy as np
import pandas as pd
from scipy.stats import norm
import math

def getBeliefwMovingObj(N, observation, transitionP, carLength=1):
    std = carLength * 1./3
    # print(N, observation, transitionP, carLength)

    # period of observation time
    timeSteps = observation.shape[0]

    # tracking hidden car one step at a time. carTrackingFrames is a collection of probMaps.
    carTrackingFrames = np.zeros((timeSteps+1, N, N))  # space holder

    # initial position: every spot has the same probability
    carTrackingFrames[0] = 1./(N*N)

    # your code

    # Convert transition probabilities into a dictionary for easy access
    transitionProbDict = {(row['X'], row['Y']): {'N': row['N'], 'E': row['E'], 'S': row['S'], 'W': row['W']}
                          for _, row in transitionP.iterrows()}

    for index, row in observation.iterrows():
        t = index + 1  # Adjust index to start from 1 for the time steps
        agentX, agentY, eDist = row['agentX'], row['agentY'], row['eDist']

        for x in range(N):
            for y in range(N):
                prob_sum = 0
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # N, E, S, W
                    prev_x, prev_y = (x + dx) % N, (y + dy) % N
                    direction = {(0, 1): 'S', (1, 0): 'W', (0, -1): 'N', (-1, 0): 'E'}[(dx, dy)]
                    transition_prob = transitionProbDict.get((prev_x, prev_y)).get(direction)
                    prob_sum += carTrackingFrames[t-1, prev_x, prev_y] * transition_prob

                dx = min(abs(agentX - x), N - abs(agentX - x))
                dy = min(abs(agentY - y), N - abs(agentY - y))
                true_dist = math.sqrt(dx**2 + dy**2)
                emission_prob = norm.pdf(eDist, true_dist, std)

                carTrackingFrames[t, x, y] = prob_sum * emission_prob

        # Normalize the probabilities for time step t
        carTrackingFrames[t] /= np.sum(carTrackingFrames[t])

    return carTrackingFrames[1:]  # Exclude the initial belief state

# No need to change the main function.
def main():
    # example: 10 20 movingCarReading10.csv transitionProb10.csv
    gridSize, reportingTime, microphoneReadingFileName, transitionProbFileName = sys.argv[1:]
    gridSize, reportingTime = int(gridSize), int(reportingTime)

    transitionP = pd.read_csv(transitionProbFileName)
    readings = pd.read_csv(microphoneReadingFileName, nrows=reportingTime)
    readings_df = pd.DataFrame(readings, columns=['agentX', 'agentY', 'eDist'])

    probMapWithTime = getBeliefwMovingObj(gridSize, readings_df, transitionP)

    mostProbableCarPosWithTime = np.zeros([reportingTime, 2])
    for t in range(reportingTime):
        mostProbableCarPosWithTime[t] = np.unravel_index(np.argmax(probMapWithTime[t], axis=None), probMapWithTime[t].shape)

    df = pd.DataFrame(mostProbableCarPosWithTime, columns=['carX', 'carY'], dtype=np.int32)
    df.to_csv("most probable location with time" + str(gridSize) + "_tillTime" + str(reportingTime) + ".csv", index=False)
    print("Most probable location (row#, column#) at time", reportingTime, ":", tuple(df.iloc[-1]))


if __name__ == '__main__':
    main()
