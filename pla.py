#this is a simple ml test
import matplotlib.pyplot as plt
import numpy as np

dataset = np.array([
((1, -0.4, 0.3), -1),
((1, -0.3, -0.1), -1),
((1, -0.2, 0.4), -1),
((1, -0.1, 0.1), -1),
((1, 0.9, -0.5), 1),
((1, 0.7, -0.9), 1),
((1, 0.8, 0.2), 1),
((1, 0.4, -0.6), 1)])
def check_error(w,dataset):
	result = None
	error = 0
	for x,s in dataset:
		x = np.array(x)#cheange tuple to array
		if int (np.sign(np.inner(w,x)))!= s:#inner product
			result = x,s
			#print result
			error+=1
	print "error=%s/%s"%(error,len(dataset))
	return result

def pla(dataset):
	w =np.zeros(3)
	while(check_error(w,dataset) is not None):
		x,s = check_error(w,dataset)
		w += s*x
		print "change finish"
	return w

w = pla(dataset)
ps = [v[0] for v in dataset]
print ps 
fig = plt.figure()
ax1 = fig.add_subplot(111)
print ax1

ax1.scatter([v[1] for v in ps[:4]], [v[2] for v in ps[:4]], s=10, c='b', marker="o", label='O')
ax1.scatter([v[1] for v in ps[4:]], [v[2] for v in ps[4:]], s=10, c='r', marker="x", label='X')
l = np.linspace(-2,2)
a,b = -w[1]/w[2], -w[0]/w[2]
ax1.plot(l, a*l + b, 'b-')
plt.legend(loc='upper left');
plt.show()
'''
for x,s in dataset
plt.plot(x.array,)
plt.ylabel('some numbers')
plt.show()
'''
