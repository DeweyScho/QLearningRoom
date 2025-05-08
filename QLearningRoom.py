#Dewey Schoenfelder
import numpy as np
import sys
from QLearning import QLearning

def main():
    # R: rows = states, cols = actions, value = reward
    R = np.array([
        [-1, -1, -1, -1,  0, -1],
        [-1, -1, -1,  0, -1, 100],
        [-1, -1, -1,  0, -1, -1],
        [-1,  0,  0, -1,  0, -1],
        [ 0, -1, -1,  0, -1, 100],
        [-1,  0, -1, -1,  0, 100]
    ])

    print("Reward matrix R:\n", R)

    qlearn = QLearning(R)
    qlearn.train(20)

    print("Learned Q matrix:\n", qlearn.Q)

if __name__ == "__main__":
    sys.exit(int(main() or 0))

