# We do some simple test for backpropagation, now have 8 weight and 2 bias
# there are two neural in network 
import math


learning_rate = 0.5
## activation function is logistric
def sigmoid(x):
    return 1/(1+ math.exp(-x))

def Error_func(input_num,target,output):
    error_value = 0
    if len(target)!= len(output):
        return false
    if input_num != len(target):
        print("wrong input, the input_num is not same with the target")
        return false

    for i in range(input_num):
        error_value+=math.pow(target[i]-output[i],2)
    return error_value/input_num

#define the neural net function

def Net_neural(weight1,weight2,x1,x2,bias):
    return weight1*x1 + weight2*x2 + bias
# Partial derivative for out_o net_o , the activation function is logistric
def Partial_outo_neto(out_o , net_o):
    return out_o*(1-out_o)



the_input = [0.05 ,0.1]
the_output = [0.01,0.99]
#weight
w = {'w1':0.15,'w2':0.2,'w3':0.25,'w4':0.3,'w5':0.40,'w6':0.45,'w7':0.5,'w8':0.55}
#bias
b ={'b1':0.35,'b2':0.60}



#hidden layer neural
# input_1
#   \
#    \w1     logistric
#     \          |
#      \         |
# b1 ---  net_h1 | out_h1 
#      /         |
#     /          |
#    /w2
#   /
# input_2


# forward pass

#hidden layer
net_h1 = Net_neural(w['w1'],w['w2'],the_input[0],the_input[1], b['b1'])
net_h2 = Net_neural(w['w3'],w['w4'],the_input[0],the_input[1], b['b1'])
print("net_h1:",net_h1 ,"net_h2",net_h2,"\n\n")
out_h1 = sigmoid(net_h1)
out_h2 = sigmoid(net_h2)
print("the two logistric func",out_h1,out_h2,"\n\n")
#output layer
net_o1 = Net_neural(w['w5'], w['w6'], out_h1, out_h2, b['b2'])
net_o2 = Net_neural(w['w7'], w['w8'], out_h1, out_h2, b['b2'])
print("net_o1:",net_o1,"net_o2",net_o2,"\n\n")
out_o1 = sigmoid(net_o1)
out_o2 = sigmoid(net_o2)
print("the output logistric func ", out_o1,out_o2,"\n\n")

E_total = Error_func(2,the_output,[out_o1,out_o2])
print("Etotal",E_total)

# backward pass

# output layer
# we want to do gradient for w5
#
# output_h1
#          \
#           \w5      logistric    
#            \           |
#             \          |
#              \         |
# bias_2    ---- net_o1  |  out_o1 ====> for E_o1
#              /         |
#             /          |
#            /w6         |
#           /
#          /
#         / 
# output_h2


# E_total  = E_o1 + E_o2
# E_o1 = 1/2 * math.pow((target_o1 - output_o1),2)
# Use Chain rule
# ∂E_total\∂w5 = ∂E_total/∂out_o1 * ∂out_o1/∂net_o1 * ∂net_o1/∂w5
## ∂E_total/∂out_o1

partial_Eo1_outo1 =  -2 *1/2 *(the_output[0]-out_o1)
## ∂out_o1/∂net_o1
partial_outo1_neto1 = Partial_outo_neto(out_o1,net_o1)
## ∂net_o1/∂w5
partial_neto1_w5 = out_h1

partial_Etotal_w5 = partial_Eo1_outo1* partial_outo1_neto1*partial_neto1_w5


##use the same function find w6 w7 w8
partial_Etotal_w6 = partial_Eo1_outo1 * partial_outo1_neto1*out_h2

#w7
partial_Eo2_outo2 = -2 * 1/2*(the_output[1] - out_o2)
partial_outo2_neto2 = Partial_outo_neto(out_o2,net_o2)
partial_neto2_w7 = out_h2

partial_Etotal_w7 = partial_Eo2_outo2 * partial_outo2_neto2 * partial_neto2_w7

#w8
partial_neto2_w8 = out_h2

partial_Etotal_w8 = partial_Eo2_outo2 * partial_outo2_neto2 * partial_neto2_w8


#
#
#
# input1 
#       \
#        \     logistric
#         \w1       |
#          \        |
#           \       |
#  bias ---- net_h1 | out_h1
#           /       |
#          /w2      |
#         /
#        /
#       /
# input2
#
#
# Use Chain rule
# ∂E_total/∂w1 = ∂E_total/∂out_h1 * ∂out_h1/∂net_h1 * ∂net_h1/∂w1
# ∂E_total/∂out_h1 = ∂E_o1/net_o1 * ∂net_o1/∂out_h1(w5) + ∂E_o2/∂out_h1
# ∂E_o1/∂net_o1 = ∂E_o1/∂out_o1 * ∂out_o1/∂net_o1


#w1
partial_Eo1_outh1 = partial_Eo1_outo1 * partial_outo1_neto1 * w['w5']
partial_Eo2_outh1 = partial_Eo2_outo2 * partial_outo2_neto2 * w['w7']

partial_Etotal_outh1 = partial_Eo1_outh1 + partial_Eo2_outh1

partial_outh1_neth1 = out_h1 * (1-out_h1)

partial_Etotal_w1 = partial_Etotal_outh1 * partial_outh1_neth1 * the_input[0]
print( partial_Etotal_w1)
#w2
partial_Etotal_w2 = partial_Etotal_outh1 * partial_outh1_neth1 * the_input[1]

#w3
partial_Eo1_outh2 = partial_Eo1_outo1 * partial_outo1_neto1 * w['w6']
partial_Eo2_outh2 = partial_Eo2_outo2 * partial_outo2_neto2 * w['w8']
partial_Etotal_outh2 = partial_Eo1_outh2 + partial_Eo2_outh2

partial_outh2_neth2 = out_h2 * (1-out_h2)
partial_Etotal_w3 = partial_Etotal_outh2 * partial_outh1_neth1 * the_input[0]
#w4
partial_Etotal_w4 = partial_Etotal_outh2 * partial_outh1_neth1 * the_input[1]

w['w1'] = w['w1'] - learning_rate * partial_Etotal_w1
w['w2'] = w['w2'] - learning_rate * partial_Etotal_w2
w['w3'] = w['w3'] - learning_rate * partial_Etotal_w3
w['w4'] = w['w4'] - learning_rate * partial_Etotal_w4
w['w5'] = w['w5'] - learning_rate * partial_Etotal_w5
w['w6'] = w['w6'] - learning_rate * partial_Etotal_w6
w['w7'] = w['w7'] - learning_rate * partial_Etotal_w7
w['w8'] = w['w8'] - learning_rate * partial_Etotal_w8

print(w)
