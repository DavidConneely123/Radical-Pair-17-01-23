import matplotlib.pyplot as plt

from RadicalPair import *
from Programs import *


def Plot_Histogram(folder_name, file_name, color = 'green', frequency_bin_width = 0.5):
    histogram_heights = np.load(folder_name + '/' + file_name + '_HISTOGRAM_HEIGHTS.npy')
    bin_centres = np.load(folder_name + '/' + file_name +  '_Bins_Centres.npy')

    plt.bar(bin_centres,histogram_heights, width=frequency_bin_width, color = color, alpha=0.5, label = f'{file_name}')
    plt.legend()

def Plot_Histogram_and_Mod_and_Modz(folder_name,file_name):
    ax,fig = plt.subplots()
    Plot_Histogram(folder_name, file_name, color='blue')
    Plot_Histogram(folder_name, file_name+'_mod', color='red')
    Plot_Histogram(folder_name, file_name+'_modz', color='green')

Plot_Histogram_and_Mod_and_Modz('Quantum Needle Testing: 24-01-23', 'A1_B0perpz_B1paraB0')
Plot_Histogram_and_Mod_and_Modz('Quantum Needle Testing: 31-01-23', 'A2_B0perpz_B1paraB0')

Plot_Histogram_and_Mod_and_Modz('Quantum Needle Testing: 24-01-23', 'A1_B0paraz_B1paraB0')
Plot_Histogram_and_Mod_and_Modz('Quantum Needle Testing: 31-01-23', 'A2_B0paraz_B1paraB0')

Plot_Histogram_and_Mod_and_Modz('Quantum Needle Testing: 24-01-23', 'A1_B0paraz_B1perpB0')
Plot_Histogram_and_Mod_and_Modz('Quantum Needle Testing: 31-01-23', 'A2_B0paraz_B1perpB0')

Plot_Histogram_and_Mod_and_Modz('Quantum Needle Testing: 24-01-23', 'A1_B0perpz_B1perpB0')
Plot_Histogram_and_Mod_and_Modz('Quantum Needle Testing: 31-01-23', 'A2_B0perpz_B1perpB0')


plt.show()


















'''
ax1, fig1 = plt.subplots()
Plot_Histogram('Quantum Needle Testing: 24-01-23', 'A1_B0perpz_B1paraB0', color = 'blue')
Plot_Histogram('Quantum Needle Testing: 24-01-23', 'A1_B0perpz_B1paraB0_mod', color = 'red')
Plot_Histogram('Quantum Needle Testing: 24-01-23', 'A1_B0perpz_B1paraB0_modz', color = 'green')
#Plot_Histogram('Quantum Needle Testing: 24-01-23', 'A1_B0paraz_B1paraB0', color = 'blue')
#Plot_Histogram('Quantum Needle Testing: 24-01-23', 'A1_B0paraz_B1perpB0', color = 'red')

ax2,fig2 = plt.subplots()
Plot_Histogram('Quantum Needle Testing: 31-01-23', 'A2_B0perpz_B1paraB0', color = 'blue')
Plot_Histogram('Quantum Needle Testing: 31-01-23', 'A2_B0perpz_B1paraB0_mod', color = 'red')
Plot_Histogram('Quantum Needle Testing: 31-01-23', 'A2_B0perpz_B1paraB0_modz', color = 'green')
#Plot_Histogram('Quantum Needle Testing: 31-01-23', 'A2_B0paraz_B1paraB0', color = 'blue')
#Plot_Histogram('Quantum Needle Testing: 31-01-23', 'A2_B0paraz_B1perpB0', color = 'red')
'''


#Plot_Histogram('Quantum Needle Testing: 24-01-23', 'A1_averaged_100directions_B0paraB1', color = 'blue')
#Plot_Histogram('Quantum Needle Testing: 24-01-23', 'A1_averaged_100directions_B0perpB1', color = 'red')


