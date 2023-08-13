import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def remove_worse(x, y):
    if len(x) != len(y):
        STOP
    #x-y plane, to right and lower is worse
    ind = []
    for i in range(len(x)):
        others = np.arange(len(x))
        others = np.delete(others, i)
        for j in others:
            if x[i] >= x[j] and y[i] <= y[j]:
                ind.append(i)
                break
    ind = np.unique(ind)
    x = np.delete(x, ind)
    y = np.delete(y, ind)
    return x, y

def order(x, y):
    if len(x) != len(y):
        STOP
    sort = np.argsort(x)
    x = x[sort]
    y = y[sort]
    return x, y

def piecewiseConstant(x,y, x_max):
    if len(x) != len(y):
        STOP
    x_new = []
    y_new = []
    for i in range(1, len(x)):
        x_new.append(x[i])
        y_new.append(y[i-1])
    if x[-1] < x_max:
        x_new.append(x_max)
        y_new.append(y[-1])
    
    x2 = []
    y2 = []
    x2.append(x[0])
    y2.append(y[0])
    for i in range(1, len(x)):
        x2.append(x_new[i-1])
        x2.append(x[i])
        y2.append(y_new[i-1])
        y2.append(y[i])
    if x2[-1] < x_max:
        x2.append(x_new[-1])
        y2.append(y_new[-1])

    x2 = np.asarray(x2)
    y2 = np.asarray(y2)
    return x2, y2

root = '/esp/'
mus_strings = ['1e-5', '1e-4', '1e-3', '1e-2', '1e-1']

right = 0

plt.figure(figsize=(8, 8))
for i in range(1,4):
    for mu_string in mus_strings:
        larm_val_T = np.load(root+'larm/run'+str(i)+'/mu='+mu_string+'/val_mean_Ts.npy')
        larm_val_acc = np.load(root+'larm/run'+str(i)+'/mu='+mu_string+'/val_accs.npy')
        
        larm_val_T, larm_val_acc = remove_worse(larm_val_T, larm_val_acc)
        larm_val_T, larm_val_acc = order(larm_val_T, larm_val_acc)
        
        right = np.amax([right, larm_val_T[-1]])
        
        plt.subplot(3, 1, i)
        plt.plot(larm_val_T, larm_val_acc, 'o-', label=mu_string)
    plt.title('larm')
    plt.xlabel('val T')
    plt.ylabel('val acc')
    plt.legend(loc='lower right')
plt.show()

plt.figure(figsize=(8, 8))
for i in range(1,4):
    for mu_string in mus_strings:
        sil_val_T = np.load(root+'cis/run'+str(i)+'/mu='+mu_string+'/val_mean_Ts.npy')
        sil_val_acc = np.load(root+'cis/run'+str(i)+'/mu='+mu_string+'/val_accs.npy')
        
        sil_val_T, sil_val_acc = remove_worse(sil_val_T, sil_val_acc)
        sil_val_T, sil_val_acc = order(sil_val_T, sil_val_acc)
        
        right = np.amax([right, sil_val_T[-1]])
        
        plt.subplot(3, 1, i)
        plt.plot(sil_val_T, sil_val_acc, 'o-', label=mu_string)
    plt.title('sil')
    plt.xlabel('val T')
    plt.ylabel('val acc')
    plt.legend(loc='lower right')
plt.show()



avgLarmAUC = 0
avgSilAUC = 0

plt.figure(figsize=(8, 8))
for i in range(1,4):
    larm_val_T = []
    larm_val_acc = []
    for mu_string in mus_strings:
        larm_val_T.extend(np.load(root+'larm/run' + str(i)+'/mu='+mu_string+'/val_mean_Ts.npy'))
        larm_val_acc.extend(np.load(root+'larm/run' + str(i)+'/mu='+mu_string+'/val_accs.npy') )
    larm_val_T, larm_val_acc = remove_worse(larm_val_T, larm_val_acc)
    larm_val_T, larm_val_acc = order(larm_val_T, larm_val_acc)
    
    sil_val_T = []
    sil_val_acc = []
    for mu_string in mus_strings:
        sil_val_T.extend(np.load(root+'cis/run'+str(i)+'/mu='+mu_string+'/val_mean_Ts.npy'))
        sil_val_acc.extend(np.load(root+'cis/run'+str(i)+'/mu='+mu_string+'/val_accs.npy') )
    sil_val_T, sil_val_acc = remove_worse(sil_val_T, sil_val_acc)
    sil_val_T, sil_val_acc = order(sil_val_T, sil_val_acc)
    
    
    larm_val_T, larm_val_acc = piecewiseConstant(larm_val_T, larm_val_acc , right)
    sil_val_T, sil_val_acc = piecewiseConstant(sil_val_T, sil_val_acc, right)
    
    avgLarmAUC += metrics.auc(larm_val_T, larm_val_acc) / 3
    avgSilAUC += metrics.auc(sil_val_T, sil_val_acc) / 3

    if i == 3:
        plt.plot(sil_val_T, sil_val_acc, 'o-', label='Our CIS (Mean AUC=%.2f)'%avgSilAUC, color='tab:orange', alpha=0.5) 
        plt.plot(larm_val_T, larm_val_acc, 'o-', label='LARM (Mean AUC=%.2f)'%avgLarmAUC, color='tab:green', alpha=0.5)  
    else:
        plt.plot(larm_val_T, larm_val_acc, 'o-', color='tab:green', alpha=0.5) 
        plt.plot(sil_val_T, sil_val_acc, 'o-', color='tab:orange', alpha=0.5)     
        
accCirc = 0.6179
TCirc = 1.502   
plt.plot(TCirc, accCirc, 'ro', markersize=30, fillstyle='none', markeredgewidth=3)

accCirc = 0.7183
TCirc = 1.5077
plt.plot(TCirc, accCirc, 'ro', markersize=30, fillstyle='none', markeredgewidth=3)

plt.xlabel('Mean T', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(loc='lower right', fontsize=20)
labels = ['Image', 'Word 1', 'Word 2', 'Word 3', 'Word 4', 'Word 5', 'Word 6', 'Word 7']
plt.xticks(range(8), labels, fontsize=10)
plt.yticks(fontsize=15)
plt.savefig('paretoAUC.png')
plt.savefig('paretoAUC.svg')
plt.show()


