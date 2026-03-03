import numpy as np
from scipy import signal

# ## 1. Convolution
#
# Use Python functions `scipy.signal.convolve2d` or `scipy.signal.convolve`. Here are the links to the function documentation.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html

# ### 1. a.  Convolve the 2D image $I$ with the 2D kernel $G$, both given below. To properly handle the borders, <u>extend the input by replicating the values, and set the output size to be the same as the input</u>.
#
# $$I = \begin{bmatrix} 5&4&0&3 \\ 6&2&1&8 \\ 7&9&4&2 \\ 8&3&6&1\end{bmatrix} \qquad G = \frac{1}{16}\begin{bmatrix} 1&2&1 \\ 2&4&2 \\ 1&2&1\end{bmatrix} \qquad $$

# In[2]:


im = np.array([[5, 4, 0, 3],
               [6, 2, 1, 8],
               [7, 9, 4, 2],
               [8, 3, 6, 1]])
conv = np.array([[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1]]) / 16

im_conv = signal.convolve2d(im, conv, "same")
print("I conv G:")
print(im_conv)


# 1. b.

conv_h = np.array([[1, 2, 1]]) / 4
conv_v = conv_h.T

# convolve with horizontal first, then vertical
im_conv_h = signal.convolve2d(im, conv_h, "same")
im_conv_h_conv_v = signal.convolve2d(im_conv_h, conv_v, "same")

print("\nConvolve with horizontal first, then vertical:")
print(im_conv_h_conv_v)

# convolve with vertical first, then horizontal
im_conv_v = signal.convolve2d(im, conv_v, "same")
im_conv_v_conv_h = signal.convolve2d(im_conv_v, conv_h, "same")

print("\nConvolve with vertical first, then horizontal:")
print(im_conv_v_conv_h)
