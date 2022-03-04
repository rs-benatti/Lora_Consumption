import matplotlib.pyplot as plt
import math
from matplotlib import cm
import numpy as np
from matplotlib.patches import Rectangle

frequencia_lora = 9.15*(10^8)
velocidade = 3*(10^8)
comprimento_onda = velocidade/frequencia_lora
n=3.3
distances = [1, 115, 350, 920, 930, 940, 1600, 1610, 2050, 4000]
pls = []
rssi = []
minTxPowerArray = []


#Sensibilidade do receptor em função de SF e BW
#Matriz com valores da sensibilidade na qual a primeira coluna BW = 500kHz, na segunda BW = 250kHz, na terceira BW = 125kHz 
#Valores provenientes do datasheet https://www.mouser.com/datasheet/2/761/sx1276-1278113.pdf
'''
[
[Srx(BW=500, SF=6), Srx(BW=500, SF=7), Srx(BW=500, SF=8), Srx(BW=500, SF=9), Srx(BW=500, SF=10), Srx(BW=500, SF=11), Srx(BW=500, SF=12)],
[Srx(BW=250, SF=6), Srx(BW=250, SF=7), Srx(BW=250, SF=8), Srx(BW=250, SF=9), Srx(BW=250, SF=10), Srx(BW=250, SF=11), Srx(BW=250, SF=12)],
[Srx(BW=125, SF=6), Srx(BW=125, SF=7), Srx(BW=125, SF=8), Srx(BW=125, SF=9), Srx(BW=125, SF=10), Srx(BW=125, SF=11), Srx(BW=125, SF=12)]
]
'''
#Srx = [[-111, -116, -119, -122, -125, -128, -130], [-115, -120, -123, -125, -128, -130, -133], [-118, -123, -126, -129, -132, -133, -136]]
Srx = [[-116, -119, -122, -125, -128, -130], [-120, -123, -125, -128, -130, -133], [-123, -126, -129, -132, -133, -136]] #SF mínimo = 7
#----------------------------------------------------- Aumento do consumo de energia--------------------------------------->>>>>>>>>>>>>>>>

def logDistance(distance):
    #pl = (20 * math.log10(4 * math.pi/comprimento_onda)) + 10 * n * math.log10(distance)
    pl = (20 * math.log10(4 * math.pi/comprimento_onda)) + 10 * n * math.log10(distance)
    return -1 * pl

def pathLoss(distance):
    return -1 * logDistance(distance)

def RSSIToConfig(RSSI, minTxPower = 0, maxTxPower = 23):
    BW = [500, 250, 125]
    SF = [7, 8, 9, 10, 11, 12]
    configMatrix = [] #Matriz que será retornada com as possíveis configurações
    for i in range(0, len(Srx)):
        for j in range(0, len(Srx[i])):
            for txPower in range(minTxPower, maxTxPower + 1):
                PL = RSSI
                if (txPower - Srx[i][j] >= PL):
                    configMatrix.append([BW[i], SF[j], txPower])
    return configMatrix


def showLossAndRSSI():
    for x in distances:
        pls.append(pathLoss(x))
        rssi.append(logDistance(x))
    plt.plot(distances, pls)
    plt.plot(distances, rssi)
    #plt.plot(distances, minTxPowerArray)
    plt.legend(["Path Loss", "RSSI", "Tx Power minimo"])
    plt.show()

def compareMatrixConsumption(configMatrix, payloadSize):
    consumptionMatrix = []
    lowestEnergy = getTransmissionEnergy(configMatrix[0][0], configMatrix[0][1], configMatrix[0][2], payloadSize)
    consumptionMatrix.append([configMatrix[0][0], configMatrix[0][1], configMatrix[0][2], lowestEnergy * 1000])
    for row in configMatrix:
        energy = getTransmissionEnergy(row[0], row[1], row[2], payloadSize) * payloadSize
        if (energy <= lowestEnergy):
            lowestEnergy = energy
            consumptionMatrix[0] = [row[0], row[1], row[2], lowestEnergy]
    return consumptionMatrix[0]


def getTimeOnAir(BW, SF, PL, preambleLength = 12, CRC = 1, header = 1, DE =0, CR = 1):
    symbolDuration = getSymbolDuration(BW, SF)
    Tpreamble = (preambleLength + 4.25) * symbolDuration
    parteSuperior = (8*PL) - (4*SF) + 28 + (16*CRC) - 20*(1-header)
    parteInferior = 4*SF - 8*DE
    ceil = math.ceil(parteSuperior/parteInferior)*(CR + 4)
    Tpayload = (8 + max(ceil, 0)) * symbolDuration
    ToA = Tpreamble + Tpayload
    return ToA

def getSymbolDuration(BW, SF):
    return ((2**SF) / BW)

def getTransmissionEnergy(BW, SF, Ptx, PL, preambleLength = 12, CRC = 1, header = 0, DE =0, CR = 1):
    return dBmToW(Ptx) * (getTimeOnAir(BW * 1000, SF, PL, preambleLength = 12, CRC = 1, header = 0, DE =0, CR = 1)/1000)

def dBmToW(Ptx):
    return (10**(Ptx/10))/1000

def plotGraph(distances, BWMatrix, SFMatrix, PtxMatrix, consumptionMatrix):    
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("RSSI (dBm)")
    ax1.set_ylabel("Ptx (dBm) e SF")
    plot1 = ax1.plot(distances, PtxMatrix, color="red")
    plot2 = ax1.plot(distances, SFMatrix, color="blue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Consumo de energia (mJ) e Largura de banda (kHz)")
    plot3 = ax2.plot(distances, consumptionMatrix, color="yellow")
    plot4 = ax2.plot(distances, BWMatrix, color="green")
    plot = plot1 + plot2 + plot3 + plot4
    ax1.legend(plot, ["Ptx otimizado", "SF otimizado", "Energia mínima para transmissão", "BW otimizado"])
    ax1.grid()
    plt.show()

def plot3D(rssiArray, SFMatrix, PtxMatrix):
    SFMatrix = np.array(SFMatrix)
    rssiArray = np.array(rssiArray)
    PtxMatrix = np.array(PtxMatrix)
    print(SFMatrix)
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))

    #===============
    #  First subplot
    #===============
    # set up the axes for the first plot
    ax = plt.subplot(projection='3d')

    surf = ax.scatter(PtxMatrix, rssiArray, SFMatrix, c=rssiArray,
                           linewidth=0, antialiased=True, depthshade=False)
    ax.set_zlim(6, 12)
    ax.set_xlabel("Ptx")
    ax.set_ylabel("Path Loss")
    ax.set_zlabel("SF")
    cax = plt.axes([0.7, 0.2, 0.01, 0.6])
    plt.colorbar(surf, cax=cax)
    '''
    #===============
    # Second subplot
    #===============
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    # plot a 3D wireframe like in the example mplot3d/wire3d_demo
    X, Y, Z = get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
    '''
    ax.set_title("Optimal SF and Ptx variation in function of Path Loss")
    plt.title("Path Loss")
    plt.show()

def plot2D(rssiArray, SFMatrix, PtxMatrix):
    SFMatrix = np.array(SFMatrix)
    rssiArray = np.array(rssiArray)
    PtxMatrix = np.array(PtxMatrix)
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle('Optimal SF and Ptx values for given Path Loss')
    axs[0].plot(rssiArray, SFMatrix)
    axs[1].plot(rssiArray, PtxMatrix)
    axs[0].set_xlim(158, 25)
    axs[1].set_xlim(158, 25)
    axs.flat[0].set(ylabel="Optimal SF")
    axs.flat[1].set(ylabel="Optimal Ptx (dBm)")
    axs.flat[1].set(xlabel="Path Loss (dBm)")
    plt.show()

def heatMap(rssiArray, SFMatrix, PtxMatrix):
    SFMatrix = np.array(SFMatrix)
    rssiArray = np.array(rssiArray)
    PtxMatrix = np.array(PtxMatrix)
    fig, ax = plt.subplots()
    sc = ax.scatter(SFMatrix, PtxMatrix, c=rssiArray)
    ax.set_ylabel('Ptx',  fontsize=14)
    ax.set_xlabel('Spreading Factor',  fontsize=14)
    ax.set_xticks([7, 8, 9, 10, 11])
    cbar = fig.colorbar(sc)
    cbar.set_label("Path Loss",  fontsize=14)
    print(f'Path Loss: {rssiArray}')
    print(f'SF: {SFMatrix}')
    plt.title('(SF, Ptx) combination for a given Path Loss',  fontsize=18)
    rect = Rectangle((8, 0), 3, 20, linestyle='dashed', facecolor='None', ec="red")
    ax.add_patch(rect)
    fig.patch.set_facecolor('#f2f2f2')
    ax.set_facecolor('#f2f2f2')
    fig.savefig('r.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()



def main():
    rssiArray = []
    for i in range(30, 152):
        print(i)
        rssiArray.append(i)
    print(rssiArray)
    consumptionMatrix = []
    PtxMatrix = []
    SFMatrix = []
    BWMatrix = []
    configMatrix = []
    '''
    #Comparação por número de bytes
    for i in range(1, 256):
        for rssiElement in rssiArray:
            config = compareMatrixConsumption(pathLossToConfig(rssiElement * -1), i)
            BWMatrix.append(config[0])
            SFMatrix.append(config[1])
            PtxMatrix.append(config[2])
            consumptionMatrix.append(config[3])
            configMatrix.append([config[0], config[1], config[2]])
        if (configMatrix != oldMatrix):
            print(f'Matriz com número de bytes {i} possui configuração diferente das outras')
            oldMatrix = configMatrix
        configMatrix = []

    '''
    for rssiElement in rssiArray:
        config = compareMatrixConsumption(RSSIToConfig(rssiElement), 3)
        BWMatrix.append(config[0])
        SFMatrix.append(config[1])
        PtxMatrix.append(config[2])
        consumptionMatrix.append(config[3])
        configMatrix.append([config[0], config[1], config[2]])
    toArduinoMatrix = str(configMatrix)
    toArduinoMatrix = toArduinoMatrix.replace("],", "},\n")
    toArduinoMatrix = toArduinoMatrix.replace("]", "}")
    toArduinoMatrix = toArduinoMatrix.replace("[", "{")
    print(toArduinoMatrix)
    #plotGraph(rssiArray, BWMatrix, SFMatrix, PtxMatrix, consumptionMatrix)
    heatMap(rssiArray, SFMatrix, PtxMatrix)

def main2():
    vetores = [[11, 23], [11, 22], [11, 21], [10, 23], [10, 22], [10, 21], [9, 23], [9, 22], [9, 21], [8, 23], [8, 22], [8, 21], [7, 23], [7, 22], [7, 21], [7, 20], [7, 19], [7, 18], [7, 17], [7, 16], [7, 15], [7, 14], [7, 13], [7, 12], [7, 11], [7, 10], [7, 9], [7, 8], [7, 7], [7, 6], [7, 5], [7, 4], [7, 3], [7, 2], [7, 1], [7, 0]]
    array = []
    for i in range(0, 36):
        array.append(round(getTimeOnAir(500, vetores[i][0], 3)/1000, 4))
    print(array)

if __name__ == "__main__":
    main()
