import numpy as np


def euclidean_distance(from_, to):
    return float(np.sqrt(np.sum((np.array(from_) - np.array(to)) ** 2)))


def manhattan_distance(from_, to):
    return float(np.sum(np.abs(np.array(from_) - np.array(to))))


def chessboard_distance(from_, to):
    return float(np.max(np.abs(np.array(from_) - np.array(to))))


total_rows = 5
total_cols = 5
center = (total_cols // 2, total_rows // 2)  # x, y

euclidean_distance_matrix = np.zeros((5, 5))
manhattan_distance_matrix = np.zeros((5, 5))
chessboard_distance_matrix = np.zeros((5, 5))


for y in range(total_rows):
    for x in range(total_cols):
        euclidean_distance_matrix[y, x] = euclidean_distance((x, y), center)
        manhattan_distance_matrix[y, x] = manhattan_distance((x, y), center)
        chessboard_distance_matrix[y, x] = chessboard_distance((x, y), center)

print("Euclidean distance matrix:")
print(np.round(euclidean_distance_matrix, 2))

print("\nManhattan distance matrix:")
print(manhattan_distance_matrix)

print("\nChessboard distance matrix:")
print(chessboard_distance_matrix)
