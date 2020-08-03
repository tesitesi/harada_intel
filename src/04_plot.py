import matplotlib.pyplot as plt 

pcAcc = [0.0537, 0.0537, 0.0537, 0.0537, 0.0537, 
0.0537, 0.0537, 0.0537, 0.0537, 0.0537]

normAcc = [0.2796, 0.3723, 0.4829, 0.5607, 0.6456, 0.7298,0.7913, 0.8380, 0.8988, 0.9346]


plt.plot(pcAcc, label='CNN with Phase Image')
plt.plot(normAcc, label='CNN Only')
plt.title('Accuracy to Training Data')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()