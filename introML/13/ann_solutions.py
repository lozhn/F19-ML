import numpy as np

X = np.array([[0, 0, 0],
              [0, 1, 0],
              [1, 0, 0],
              [0, 0, 1],
              [1, 0, 1],
              [1, 1, 1]])

W1=[[-4.9981605, -1.07977764, -4.63107903, 0.77975539],
    [-5.23518082, 2.9894962, 2.09285688, -2.3807279, ],
    [1.86235711, -0.05955762, 0.73450118, 1.29146894]]

W2=[[-7.35383386, 5.97548867, 0.46500659, 0.93833849, 0.95093351], 
    [-2.50376915, 1.49435331, -0.25614277, 0.97663961, 0.82391926], 
    [4.11950385, -2.6483043, 0.21555922, -1.6368795, -1.30915197], 
    [3.06518371, -2.46303259, -0.20889594, -1.0895797, -0.84178262]]

W3=[[10.07770919], 
    [-7.2610922], 
    [-0.108012], 
    [-1.95695342], 
    [-1.69213446]]

# TO DO ---- 5 POINTS --------------------- Network structure --------------------
# Given network weights, understand what is the number of layers in the ANN?
# Now many neurons in the layers the network has?
# layers: ______
# neurons: layer 1: ________ ...
# -----------------------------------------------------

# TO DO ---- 10 POINTS --------------------- prediction --------------------
# Implement forvard propogation to calculate the prediction. Use sigmoid for activation function.
def predict(x):
    pass
# Solution ----------------------------------------------------
def predict(x):
    l1 = 1 / (1 + np.exp(-(np.dot(x, W1))))
    l2 = 1 / (1 + np.exp(-(np.dot(l1, W2))))
    l3 = 1 / (1 + np.exp(-(np.dot(l2, W3))))
    return l3
# -----------------------------------------------------

y_pred = predict(X)
print(y_pred)

# TO DO ---- 5 POINTS --------------------- logical function --------------------
# Considering input and output agruments as binary values.
# Based on the output, which logical function the network implements?
# Solution ----------------------------------------------------
# x[0] xor x[1]
# -----------------------------------------------------
