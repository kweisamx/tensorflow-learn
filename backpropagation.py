# We do some simple test for backpropagation, now have 8 weight and 2 bias
# there are two neural in network 
import math


## activation function is logistric
def sigmoid(x):
    return 1/(1+ math.exp(-x))

def Error_func(input_num,target,output):
    error_value = 0
    print target,output
    if len(target)!= len(output):
        return false
    if input_num != len(target):
        print "wrong input, the input_num is not same with the target"
        return false



    for i in range(input_num):
        error_value+=math.pow(target[i]-output[i],2)
    return error_value/input_num

learing_rate = 0.5

the_input = {'i1':0.05 ,'i2':0.1}
the_output = {'o1':0.01,'o2':0.99}
#weight
w = {'w1':0.15,'w2':0.2,'w3':0.25,'w4':0.3,'w5':0.40,'w6':0.45,'w7':0.5,'w8':0.55}
#bias
b ={'b1':0.35,'b2':0.60}



# forward pass

#hidden layer
h1 = w['w1'] * the_input['i1'] + w['w2'] * the_input['i2'] + b['b1']
h2 = w['w3'] * the_input['i1'] + w['w4'] * the_input['i2'] + b['b1']
print "h1:",h1 ,"h2",h2
print "the two logistric func",sigmoid(h1),sigmoid(h2)
#output layer
o1 = w['w5'] * sigmoid(h1) + w['w6'] * sigmoid(h1) + b['b2']
o2 = w['w7'] * sigmoid(h2) + w['w8'] * sigmoid(h2) + b['b2']
print "o1:",o1,"o2",o2
print "the output logistric func ", sigmoid(o1),sigmoid(o2)
E_total = Error_func(2,[the_output['o1'],the_output['o2']],[sigmoid(o1),sigmoid(o2)])
print E_total

# backward pass

#E_total  = E_o1 + E_o2

#E_o1 = 1/2 * math.pow((target_o1 - output_o1),2)


