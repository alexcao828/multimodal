import numpy as np
import matplotlib.pyplot as plt

root = '/esp/'

accStarLARM = 0.6179
TStarLARM = 1.502

accStarSIL = 0.7183
TStarSIL = 1.5077

larm_val_T = []
larm_val_acc = []
mus_strings = ['1e-5', '1e-4', '1e-3', '1e-2', '1e-1']
for mu_string in mus_strings:
    larm_val_T = np.load(root+'larm/run1/mu='+mu_string+'/val_mean_Ts.npy')
    larm_val_acc = np.load(root+'larm/run1/mu='+mu_string+'/val_accs.npy')
    check = np.where((larm_val_T == TStarLARM) & (larm_val_acc ==  accStarLARM))[0]
    if len(check) == 1:
        muStarLARM = mu_string
        epochStarLARM = check[0]


sil_val_T = []
sil_val_acc = []
mus_strings = ['1e-5', '1e-4', '1e-3', '1e-2', '1e-1']
for mu_string in mus_strings:
    sil_val_T = np.load(root+'cis/run1/mu='+mu_string+'/val_mean_Ts.npy')
    sil_val_acc = np.load(root+'cis/run1/mu='+mu_string+'/val_accs.npy')
    check = np.where((sil_val_T == TStarSIL) & (sil_val_acc ==  accStarSIL))[0]
    if len(check) == 1:
        muStarSIL = mu_string
        epochStarSIL = check[0]






valAllTsLARM = np.load('/esp/larm/run1/mu='+str(muStarLARM)+'/val_all_Ts.npy')[epochStarLARM]
valAllTsSIL = np.load('/esp/cis/run1/mu='+str(muStarSIL)+'/val_all_Ts.npy')[epochStarSIL]
    
plt.style.use('seaborn-deep')
plt.figure(figsize=(8, 8))
plt.hist([valAllTsSIL, valAllTsLARM], bins=12, range=(0,5), label=['Our CIS', 'LARM'], color=['tab:orange', 'tab:green'])
labels = ['Image', 'Word 1', 'Word 2', 'Word 3', 'Word 4', 'Word 5']
plt.xticks(range(6), labels, fontsize=15)
labels = ['0%', '10%', '20%', '30%', '40%', '50%']
plt.yticks(range(0, 6000, 1000), labels, fontsize=15)
plt.xlabel('T', fontsize=20)
plt.ylabel('Proportion of samples', fontsize=20)
plt.legend(loc='upper right', fontsize=15)
plt.savefig('histT.eps')
plt.show()

























