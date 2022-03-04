import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import statistics as st
import seaborn as sns

vetores = ["[11, 23]", "[11, 22]", "[11, 21]", "[10, 23]", "[10, 22]", "[10, 21]", "[9, 23]", "[9, 22]", "[9, 21]", "[8, 23]", "[8, 22]", "[8, 21]", "[7, 23]", "[7, 22]", "[7, 21]", "[7, 20]", "[7, 19]", "[7, 18]", "[7, 17]", "[7, 16]", "[7, 15]", "[7, 14]", "[7, 13]", "[7, 12]", "[7, 11]", "[7, 10]", "[7, 9]", "[7, 8]", "[7, 7]", "[7, 6]", "[7, 5]", "[7, 4]", "[7, 3]", "[7, 2]", "[7, 1]", "[7, 0]"]
vetoresList = [[11, 23], [11, 22], [11, 21], [10, 23], [10, 22], [10, 21], [9, 23], [9, 22], [9, 21], [8, 23], [8, 22], [8, 21], [7, 23], [7, 22], [7, 21], [7, 20], [7, 19], [7, 18], [7, 17], [7, 16], [7, 15], [7, 14], [7, 13], [7, 12], [7, 11], [7, 10], [7, 9], [7, 8], [7, 7], [7, 6], [7, 5], [7, 4], [7, 3], [7, 2], [7, 1], [7, 0]]
vetores.reverse()
amostraTtx = [[], [], [], [], []]
amostraPtx = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
print(len(amostraPtx))
integralSum = [0] * 36
TtxArray = [0] * 36
tempTx = []
for i in range(0, 36):
    file = pd.read_csv(f'testeFormFiles/vetor[{i}].txt', sep=' ')
    criteria = file[file.iloc[:, 1] >= 30.0]
    newPicoIndex = 0
    picoLastName = 0
    criteriaLastName = criteria.iloc[-1].name
    picoCounter = 0
    while (picoLastName != criteriaLastName):
        picoCounter += 1
        pico = criteria[criteria.iloc[:, 0] < criteria.iloc[newPicoIndex, 0] + 900]
        picoTemp = pico.iloc[newPicoIndex:]
        Ttx = (picoTemp.iloc[-1, 0] - picoTemp.iloc[0, 0]) / 10
        TtxArray[i] += Ttx
        tempTx.append(Ttx / 100)
        if (i <= 2):
            amostraTtx[0].append(Ttx * 10)
        if (i > 2 and i <= 5):
            amostraTtx[1].append(Ttx * 10)
        if (i > 5 and i <= 8):
            amostraTtx[2].append(Ttx * 10)
        if (i > 8 and i <= 11):
            amostraTtx[3].append(Ttx * 10)
        if (i > 11 and i <= 14):
            amostraTtx[4].append(Ttx * 10)
        for current in list(picoTemp.iloc[:, 1]):
            amostraPtx[vetoresList[i][1]].append(int(current))
        integral = integrate.trapz(picoTemp.iloc[:, 1], picoTemp.iloc[:, 0])/360000
        integralSum[i] += integral
        newPicoIndex = len(pico.index)
        picoLastName = pico.iloc[-1].name
    tempTx = []
    TtxArray[i] = Ttx/picoCounter
    integralSum[i] = integralSum[i]/picoCounter
x = np.arange(36)

rssi = np.arange(116, 152).tolist()
rssi.reverse()
theoryToA = [0.1198, 0.1198, 0.1198, 0.0599, 0.0599, 0.0599, 0.03, 0.03, 0.03, 0.0175, 0.0175, 0.0175, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088]     
fig, ax1 = plt.subplots()
#interpolate_x = np.linspace(0, 35, 1000)
#interpolate_t = interpolate.interp1d(x, TtxArray, kind="nearest")(interpolate_x)
line1, = ax1.plot(rssi, TtxArray, '*', label="ToA medido (s)")
line2, = ax1.plot(rssi, integralSum, label="Consumo (mAh)")
line3, = ax1.plot(rssi, theoryToA, label="ToA teórico")
#line4, = ax1.plot(interpolate_x, interpolate_t, label="interpolate", color="tab:red")

#ax1.set_ylabel("Tempo e transmissão (s), consumo (mAh)")
ax1.set_ylim([0, 0.13])

def lower2upper(x_lower):
    return x_lower + 116

def upper2lower(x_upper):
    return x_upper - 116


secax = ax1.secondary_xaxis('top', functions=(upper2lower, lower2upper))
secax.set_xlabel('[SF, PTx]')
secax.tick_params(axis='x', rotation=70)
#secax.set_xticklabels(vetores, fontdict=None, minor=False)


'''
plt.plot(x, TtxArray, label="Tempo de transmissão (s)")
plt.plot(x, integralSum, label="Consumo (mAh)")
'''
sns.set_theme()
ax1.set_xticks(rssi, rssi)
ax1.set_xlabel('Path Loss')
secax.set_xticks(x)
secax.set_xticklabels(vetores, fontdict=None, minor=False)
secax.tick_params(axis='x', rotation=70)
title = plt.title("Power consumption and ToA")
title.set_position([.5, 1.2])
plt.legend([line1, line2, line3], ['Measured ToA (s)', 'Measured Power Consumption (mAh)', "Estimated ToA (s)"], loc='best')
plt.grid()
fig.patch.set_facecolor('#f2f2f2')
ax1.set_facecolor('#f2f2f2')
fig.savefig('w.png', facecolor=fig.get_facecolor(), edgecolor='none')
plt.show()
fig, ax2 = plt.subplots()
ax2.set_xlim((min(amostraTtx[4]) - 1 , max(amostraTtx[0])+ 5))
ax2.set(xlabel='Transmission time (ms)')
sns.histplot(amostraTtx[0], ax=ax2, discrete=True, color="tab:blue")
sns.histplot(amostraTtx[1], ax=ax2, discrete=True, color="tab:green")
sns.histplot(amostraTtx[2], ax=ax2, discrete=True, color="tab:red")
sns.histplot(amostraTtx[3], ax=ax2, discrete=True, color="tab:purple")
sns.histplot(amostraTtx[4], ax=ax2, discrete=True, color="#91cbca")


'''
textstr = f'Mean \u2248 {st.mean(amostraTtx[0])}\nStandard Deviation \u2248 {st.pstdev(amostraTtx[0])}'
props = dict(boxstyle='round', facecolor='gray', alpha=0.5)
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)
'''
plt.title('Distributions for Different Spreading Factors')
plt.legend(['SF11', 'SF10', "SF9", 'SF8', 'SF7'], loc='best')

plt.show()

meanCurrent = []
ptxs = np.arange(0, 24)
for i in range(0, len(amostraTtx)):
    textstr = f'SF{11-i}\nMean \u2248 {st.mean(amostraTtx[i])}\nStandard Deviation \u2248 {st.pstdev(amostraTtx[i])}'
    print(textstr)
for i in range(0, len(amostraPtx)):
    textstr = f'Ptx{i}\nMean \u2248 {st.mean(amostraPtx[i])}\nStandard Deviation \u2248 {st.pstdev(amostraPtx[i])}'
    print(textstr)
    meanCurrent.append(st.mean(amostraPtx[i]))
plt.plot(ptxs, meanCurrent)
plt.show()


'''
n, bins, patches = plt.hist(amostraTtx, bins=15)
plt.xticks(np.arange(min(amostraTtx), max(amostraTtx)+10, 1))
plt.show()
'''
