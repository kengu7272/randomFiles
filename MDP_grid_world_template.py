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

    # Perform policy evaluation for the specified number of iterations
    for _ in range(duration):
        # Iterate over all states in the grid world
        for i in range(rows):
            for j in range(cols):
                # Skip non-accessible spot and terminal states
                if (i, j) == (1, 1) or (i, j) == (2, 3) or (i, j) == (1, 3):
                    continue

                # Get the action from the policy for the current state
                action = policy[i][j]

                # directions
                upRow, upCol = (min((i + 1), rows - 1), j)
                rightRow, rightCol = (i, min((j + 1), cols - 1))
                leftRow, leftCol = (i, max(0, (j - 1)))
                downRow, downCol = (max(0, (i - 1)), j)

                # checks if any accesses inacessible state and bounces back if so
                if (upRow, upCol) == (1, 1):
                    (upRow, upCol) = (i, j)
                elif (rightRow, rightCol) == (1, 1):
                    (rightRow, rightCol) = (i, j)
                elif (leftRow, leftCol) == (1, 1):
                    (leftRow, leftCol) = (i, j)
                elif (downRow, downCol) == (1, 1):
                    (downRow, downCol) = (i, j)
                
                # sets different reward if terminal state
                upReward = rightReward = downReward = leftReward = reward

                if (upRow, upCol) == (2, 3):
                    upReward = 1
                elif (upRow, upCol) == (1, 3):
                    upReward = -1

                if (rightRow, rightCol) == (2, 3):
                    rightReward = 1
                elif (rightRow, rightCol) == (1, 3):
                    rightReward = -1
                
                if (downRow, downCol) == (2, 3):
                    downReward = 1
                elif (downRow, downCol) == (1, 3):
                    downReward = -1

                if (leftRow, leftCol) == (2, 3):
                    leftReward = 1
                elif (leftRow, leftCol) == (1, 3):
                    leftReward = -1

                # Calculate the row and column indices of the next state based on the action
                if action == 1:  # Up
                    expectedValue = (
                        transitionProb * (upReward + gamma * val[upRow][upCol]) +
                        leftProb * (leftReward + gamma * val[leftRow][leftCol]) +
                        rightProb * (rightReward + gamma * val[rightRow][rightCol])
                    )
                elif action == -1:  # Down   
                    expectedValue = (
                        transitionProb * (downReward + gamma * val[downRow][downCol]) +
                        leftProb * (leftReward + gamma * val[leftRow][leftCol]) +
                        rightProb * (rightReward + gamma * val[rightRow][rightCol])
                    )                   
                elif action == 2:  # Right  
                    expectedValue = (
                        transitionProb * (rightReward + gamma * val[rightRow][rightCol]) +
                        leftProb * (upReward + gamma * val[upRow][upCol]) +
                        rightProb * (downReward + gamma * val[downRow][downCol])
                    )          
                elif action == -2:  # Left
                    expectedValue = (
                        transitionProb * (leftReward + gamma * val[leftRow][leftCol]) +
                        leftProb * (downReward + gamma * val[downRow][downCol]) +
                        rightProb * (upReward + gamma * val[upRow][upCol])
                    )

                # Update the new value function with the expected value
                newVal[i][j] = expectedValue

        # Update the value function with the new values for the next iteration
        val = np.copy(newVal)

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
    leftProb = rightProb = (1 - prob) / 2

    # vOpt array
    val = np.zeros([3, 4], dtype=float)

    # working value array
    newVal = np.copy(val)

    for _ in range(duration):
        for i in range(rows):
            for j in range(cols):
                if (i, j) == (1, 1) or (i, j) == (2, 3) or (i, j) == (1, 3):
                    continue

                # directions
                upRow, upCol = (min((i + 1), rows - 1), j)
                rightRow, rightCol = (i, min((j + 1), cols - 1))
                leftRow, leftCol = (i, max(0, (j - 1)))
                downRow, downCol = (max(0, (i - 1)), j)

                # checks if any accesses inacessible state and bounces back if so
                if (upRow, upCol) == (1, 1):
                    (upRow, upCol) = (i, j)
                elif (rightRow, rightCol) == (1, 1):
                    (rightRow, rightCol) = (i, j)
                elif (leftRow, leftCol) == (1, 1):
                    (leftRow, leftCol) = (i, j)
                elif (downRow, downCol) == (1, 1):
                    (downRow, downCol) = (i, j)

                # sets different reward if terminal state
                upReward = rightReward = downReward = leftReward = reward

                if (upRow, upCol) == (2, 3):
                    upReward = 1
                elif (upRow, upCol) == (1, 3):
                    upReward = -1

                if (rightRow, rightCol) == (2, 3):
                    rightReward = 1
                elif (rightRow, rightCol) == (1, 3):
                    rightReward = -1
                
                if (downRow, downCol) == (2, 3):
                    downReward = 1
                elif (downRow, downCol) == (1, 3):
                    downReward = -1

                if (leftRow, leftCol) == (2, 3):
                    leftReward = 1
                elif (leftRow, leftCol) == (1, 3):
                    leftReward = -1

                # find expected values of every action  
                upVal = (
                    prob * (upReward + gamma * val[upRow][upCol]) +
                    leftProb * (leftReward + gamma * val[leftRow][leftCol]) +
                    rightProb * (rightReward + gamma * val[rightRow][rightCol])
                )
        
                rightVal = (
                    prob * (rightReward + gamma * val[rightRow][rightCol]) +
                    leftProb * (upReward + gamma * val[upRow][upCol]) +
                    rightProb * (downReward + gamma * val[downRow][downCol])
                )
        
                downVal = (
                    prob * (downReward + gamma * val[downRow][downCol]) +
                    leftProb * (leftReward + gamma * val[leftRow][leftCol]) +
                    rightProb * (rightReward + gamma * val[rightRow][rightCol])
                )
        
                leftVal = (
                    prob * (leftReward + gamma * val[leftRow][leftCol]) +
                    leftProb * (downReward + gamma * val[downRow][downCol]) +
                    rightProb * (upReward + gamma * val[upRow][upCol])
                )
                
                maxVal = max(upVal, rightVal, downVal, leftVal)
                newVal[i][j] = maxVal

                # ordered this way so that up > right > down > left
                if upVal == maxVal:
                    policy[i][j] = 1
                elif rightVal == maxVal:
                    policy[i][j] = 2
                elif downVal == maxVal:
                    policy[i][j] = -1
                elif leftVal == maxVal:
                    policy[i][j] = -2

        val = np.copy(newVal)

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
