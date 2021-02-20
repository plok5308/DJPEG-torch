import numpy as np
import math
import torch

#caclulate DCT basis
def cal_scale(p,q):
	if p==0:
		ap = 1/(math.sqrt(8))
	else:
		ap = math.sqrt(0.25)
	if q==0:
		aq = 1/(math.sqrt(8))
	else:
		aq = math.sqrt(0.25)

	return ap,aq

def cal_basis(p,q):
	basis = np.zeros((8,8))
	ap,aq = cal_scale(p,q)
	for m in range(0,8):
		for n in range(0,8):
			basis[m,n] = ap*aq*math.cos(math.pi*(2*m+1)*p/16)*math.cos(math.pi*(2*n+1)*q/16)

	return basis

def load_DCT_basis_64():
	basis_64 = np.zeros((8,8,64))
	idx = 0
	for i in range(8):
		for j in range(8):
			basis_64[:,:,idx] = cal_basis(i,j)
			idx = idx + 1
	return basis_64

def load_DCT_basis_torch():
    DCT_basis_64 = load_DCT_basis_64()
    np_basis = np.zeros((64, 1, 8, 8)) #outchannel, inchannel, height, width
    for i in range(64):
        np_basis[i,0,:,:] = DCT_basis_64[:,:,i]

    torch_basis = torch.from_numpy(np_basis)
    return torch_basis
