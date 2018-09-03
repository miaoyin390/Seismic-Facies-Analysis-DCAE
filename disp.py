import numpy as np
import matplotlib.pyplot as plt


def Wiggle(data, lWidth=0.1):
    sampleNum, traceNum = np.shape(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(traceNum):
        traceData = data[:,i]
        maxVal = np.amax(traceData)
        ax.plot(i+traceData/maxVal, [j for j in range(sampleNum)], color='black', linewidth=lWidth)
        for a in range(len(traceData)):
            if(traceData[a] < 0):
                traceData[a] = 0
        ax.fill(i+traceData/maxVal, [j for j in range(sampleNum)], 'k', linewidth=0)
    ax.axis([0,traceNum,sampleNum,0])
    plt.show()

if __name__ == '__main__':
    data = np.load('pre_data_24points.npy')
    disp_data = data[100,:,0:24]
    print(np.shape(disp_data))
    #Wiggle(disp_data.T)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(disp_data[10,:], [i for i in range(24)])
    plt.show()

