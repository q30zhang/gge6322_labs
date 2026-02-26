import numpy as np


width = 640  # equals to number of columns
height = 480  # equals to number of rows


# convert coordinates to indices
x1, y1 = (38, 52)
x2, y2 = (592, 241)
x3, y3 = (33, 0)

def coord_to_index(x, y):
    return x + y * width

index_1 = coord_to_index(x1, y1)
index_2 = coord_to_index(x2, y2)
index_3 = coord_to_index(x3, y3)

print("Coordinates to indices:")
print(f"Coordinates (x, y) = ({x1}, {y1}) is converted to index i = {index_1}")
print(f"Coordinates (x, y) = ({x2}, {y2}) is converted to index i = {index_2}")
print(f"Coordinates (x, y) = ({x3}, {y3}) is converted to index i = {index_3}")


# convert indices to coordinates
index_4 = 8092
index_5 = 24061
index_6 = 38190

def index_to_coord(index):
    return index % width, index // width  # x, y

x4, y4 = index_to_coord(index_4)
x5, y5 = index_to_coord(index_5)
x6, y6 = index_to_coord(index_6)

print("\nIndices to coordinates:")
print(f"Index i = {index_4} is converted to coordinates (x, y) = ({x4}, {y4})")
print(f"Index i = {index_5} is converted to coordinates (x, y) = ({x5}, {y5})")
print(f"Index i = {index_6} is converted to coordinates (x, y) = ({x6}, {y6})")


# verification with a random image
im = (np.random.random(size=(height, width)) * 256).astype(int)
im_1d = im.reshape(width * height)
assert im[y1, x1] == im_1d[index_1]
assert im[y2, x2] == im_1d[index_2]
assert im[y3, x3] == im_1d[index_3]
assert im[y4, x4] == im_1d[index_4]
assert im[y5, x5] == im_1d[index_5]
assert im[y6, x6] == im_1d[index_6]
