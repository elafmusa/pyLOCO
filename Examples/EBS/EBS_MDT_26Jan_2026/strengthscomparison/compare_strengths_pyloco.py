# compare strengths before and after realignment

import numpy as np
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt

# load strengths file before realignment 
def read_str_file(file):
    str_dict = {}
    quad= []
    sext= []
    oct = []
    hst = []
    vst = []
    sqp = []

    with open(file, 'r') as f:
        ls = f.readlines()
        for l in ls:
            if l[0] != '#':
                ll = l.split(':')
                name = ll[0]
                str = float(ll[1][0:-1])
    #             print(f'{name} has str {str}')
                str_dict[name] = str

    for k, v in str_dict.items():
        if 'm-q' in k:
            quad.append(v)
        # if 'm-dq' in k:
        #     quad.append(v)
        if 'm-s' in k:
            sext.append(v)
        if 'm-o' in k:
            oct.append(v)
        if 'hst' in k:
            hst.append(v)
        if 'vst' in k:
            vst.append(v)
        if 'sqp' in k:
            sqp.append(v)

    str_dict['quad'] = np.array(quad)
    str_dict['sext'] = np.array(sext)
    str_dict['oct'] = np.array(oct)
    str_dict['hst'] = np.array(hst)
    str_dict['vst'] = np.array(vst)
    str_dict['sqp'] = np.array(sqp)

    return str_dict

def read_des_str_file(file):
    str_dict = {}
    quad= []
    sext= []
    oct = []
    hst = []
    vst = []
    sqp = []

    with open(file, 'r') as f:
        ls = f.readlines()
        for l in ls:
            ll = l.split(',')
            name = ll[0]
            str = float(ll[1][0:-1])
#             print(f'{name} has str {str}')
            str_dict[name] = str

    for k, v in str_dict.items():
        if 'm-q' in k:
            quad.append(v)
        #if 'm-dq' in k:
        #    quad.append(v)
        if 'm-s' in k:
            sext.append(v)
        if 'm-o' in k:
            oct.append(v)
        if 'hst' in k:
            hst.append(v)
        if 'vst' in k:
            vst.append(v)
        if 'sqp' in k:
            sqp.append(v)

    str_dict['quad'] = np.array(quad)
    str_dict['sext'] = np.array(sext)
    str_dict['oct'] = np.array(oct)
    str_dict['hst'] = np.zeros(384)
    str_dict['vst'] = np.zeros(288)
    str_dict['sqp'] = np.zeros(288)

    return str_dict

des = read_des_str_file('/operation/beamdyn/optics/sr/S28F_all_BM_25Aug2024/DesignStrengths.csv')

# exit()

# bef = read_str_file('/machfs/MDT/2026/2026_01_18/strengthscomparison/pyLOCO_correction.ts')
bef = read_str_file('/machfs/MDT/2026/2026_01_18/strengthscomparison/DESY_MDT_pyLOCO_C2more.ts')
# aft = read_str_file('/machfs/MDT/2026/2026_01_18/strengthscomparison/restart_after_bba_optics3_LT5DA1_cycled_70perc.ts')
aft = read_str_file('/machfs/MDT/2026/2026_01_18/strengthscomparison/operation_after_restart.ts')


fig, ax = plt.subplots(nrows=6, figsize=(10,8), dpi=100)
plt.subplots_adjust(hspace=0.5)
for c, k in enumerate(['quad', 'sext', 'oct', 'hst', 'vst', 'sqp']):
    b = bef[k]-des[k]
    a = aft[k]-des[k]
    ax[c].plot(b, label=f'pyloco(Jan 2025), std= {np.std(b):2.2e}')
    ax[c].plot(a, label=f'FILO  (Jan 2026), std= {np.std(a):2.2e}')
    ax[c].set_ylabel(k)
    ax[c].legend(loc='lower right')

plt.show()
