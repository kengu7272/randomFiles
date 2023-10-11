# AI Project 2 - MDP
# Grid world - part 1 policy evaluation; part 2 value iteration
import pandas as pd
import numpy as np
import sys


def policyEvaluation(policy, reward, gamma, duration):
    # assume the gridWorld (3 rows x 4 cols) is just like the one outlined in Figure 17.1
    # reward at (2,3) is 1, (1,3) is -1; (1,1) is not accessible; (0,0) is the start.
    # transition probability is 0.8 to the intended direction, 0.1 to the left, 0.1 to the right.
    #
    print("policy:\n", policy, "\nreward, gamma, duration:\n", reward, gamma, duration)
    val = np.zeros(policy.shape)  # space hold
    # your code
    rows = 3
    cols = 4
    transitionProb = 0.8
    leftProb = 0.1
    rightProb = 0.1

    # Initialize a new value function to store updated values
    newVal = np.zeros((rows, cols))

    # Set expected rewards to terminal states
    newVal[2][3] = 1
    newVal[1][3] = -1

    # Perform policy evaluation for the specified number of iterations
    for i in range(duration):
        # Iterate over all states in the grid world
        for j in range(rows):
            for k in range(cols):
                # Skip non-accessible spot and terminal states
                if (j, k) == (1, 1) or (j, k) == (2, 3) or (j, k) == (1, 3):
                    continue

                # Get the action from the policy for the current state
                action = policy[j][k]

                # Calculate the row and column indices of the next state based on the action
                if action == 1:  # Up
                    nextState = (max(j - 1, 0), k)
                elif action == -1:  # Down
                    nextState = (min(j + 1, rows - 1), k)
                elif action == 2:  # Right
                    nextState = (j, min(k + 1, cols - 1))
                elif action == -2:  # Left
                    nextState = (j, max(k - 1, 0))

                # Calculate the expected value for the current state
                expectedValue = (
                    transitionProb * (reward + gamma * val[nextState[0]][nextState[1]]) +
                    leftProb * (reward + gamma * val[j][max(k - 1, 0)]) +
                    rightProb * (reward + gamma * val[j][min(k + 1, cols - 1)])
                )

                # Update the new value function with the expected value
                newVal[j][k] = expectedValue

        # Update the value function with the new values for the next iteration
        val = newVal

    return val


def valueIteration(reward, gamma, prob, duration):
    # assume the gridWorld (3 rows x 4 cols) is just like the one outlined in Figure 17.1
    # reward at (2,3) is 1, (1,3) is -1; (1,1) is not accessible; (0,0) is the start.
    #
    print("\nreward, gamma, prob, duration:\n", reward, gamma, prob, duration)
    policy = np.ones([3, 4], dtype=int)  # space hold
    # your code
    rows = 3
    cols = 4

    # probabilities of not going intended direction (2 directions moving right angles from intended)
    leftProb = (1 - prob) / 2
    rightProb = leftProb

    # vOpt array
    v = np.zeros([3, 4], dtype=float)

    # rewards array to calculate reward based on action
    rewards = np.full((3, 4), reward)

    # these terminal states are the only ones without the same value
    rewards[2][3] = 1
    rewards[1][3] = -1

    # make blocked off square really low so that it never recommends it
    rewards[1][1] = -1000

    # working value array
    newV = np.copy(v)

    for _ in range(duration):
        for i in range(rows):
            for j in range(cols):
                # find expected values of every action  
                upVal = (
                    prob * (rewards[min(rows - 1, i + 1)][j] + gamma * v[i][j]) +
                    leftProb * (rewards[i][max(j - 1, 0)] + gamma * v[i][j]) +
                    rightProb * (rewards[i][min(cols - 1, j + 1)] + gamma * v[i][j])
                )
        
                rightVal = (
                    prob * (rewards[i][min(cols - 1, j + 1)] + gamma * v[i][j]) +
                    leftProb * (rewards[min(rows - 1, i + 1)][j] + gamma * v[i][j]) +
                    rightProb * (rewards[max(0, i - 1)][j] + gamma * v[i][j])
                )
        
                downVal = (
                    prob * (rewards[max(0, i - 1)][j] + gamma * v[i][j]) +
                    leftProb * (rewards[i][min(cols - 1, j + 1)] + gamma * v[i][j]) +
                    rightProb * (rewards[i][max(j - 1, 0)] + gamma * v[i][j])
                )
        
                leftVal = (
                    prob * (rewards[i][max(j - 1, 0)] + gamma * v[i][j]) +
                    leftProb * (rewards[max(0, i - 1)][j] + gamma * v[i][j]) +
                    rightProb * (rewards[min(rows - 1, i + 1)][j] + gamma * v[i][j])
                )

                print(f"At {i},{j}, up:{upVal}, right:{rightVal}, down:{downVal}, left:{leftVal}")
                
                maxVal = max(upVal, rightVal, downVal, leftVal)
                newV[i][j] = maxVal

                # ordered this way so that up > right > down > left
                if upVal == maxVal:
                    policy[i][j] = 1
                elif rightVal == maxVal:
                    policy[i][j] = 2
                elif downVal == maxVal:
                    policy[i][j] = -1
                elif leftVal == maxVal:
                    policy[i][j] = -2

        v = newV

    return policy


# No need to change the main function.
def main():
    part, reward, arg3 = sys.argv[1:]
    gamma = 0.95
    duration = 50
    if part == "1":
        policyFileName = arg3
        policyData = pd.read_csv(policyFileName, header=None)
        policy = policyData.to_numpy(dtype=int)
        policy = policy[::-1]  # flip the rows to match the setup
        values = policyEvaluation(policy, float(reward), gamma, duration)
        print("The expected utility of policy given in " + policyFileName +
              " after", duration, "iterations :")
        print(values[::-1])  # flip the rows to match the setup
    elif part == "2":
        prob = float(arg3)
        print("Optimal policy after", duration, "iterations :")
        policy = valueIteration(float(reward), gamma, prob, duration)
        print(policy[::-1])  # flip the rows to match the setup
    else:
        print("arg error")

    print("\ndone!")


if __name__ == '__main__':
    main()
