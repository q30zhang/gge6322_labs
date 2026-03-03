import numpy as np


A = np.array([[1, 1, 1, 0],
              [0, 1, 0, 0],
              [1, 1, 1, 0],
              [0, 0, 0, 0]], dtype=np.bool)
B = np.array([[0, 0, 0, 0],
              [0, 1, 1, 0],
              [0, 1, 1, 1],
              [0, 1, 0, 1]], dtype=np.bool)

# union
A_union_B = np.logical_or(A, B)
print("A union B:")
print(A_union_B)

# intersection
A_intersect_B = np.logical_and(A, B)
print("\nA intersect B:")
print(A_intersect_B)

# reflection of B about the origin
B_reflection = B[::-1, ::-1]
# alternatively
B_reflection2 = np.rot90(np.rot90(B))
assert (B_reflection2 == B_reflection).all()
print("\nB reflection:")
print(B_reflection)

# A complement
A_complement = np.logical_not(A)
print("\nA complement:")
print(A_complement)

# A\B, set difference between A and B
A_setdiff_B = np.logical_and(A, np.logical_not(B))
print("\nA\\B")
print(A_setdiff_B)
