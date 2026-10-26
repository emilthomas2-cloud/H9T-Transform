"""Controlled exploratory H9T vs DCT vs Haar tests. Run: python followup_v04.py"""
import os, csv, zlib, time
from pathlib import Path
import numpy as np
from scipy.fft import dct, idct
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from skimage import data, color
OUT=Path(__file__).parent
N=9; SIDE=252
A=np.zeros((9,9));A[0,:]=1/3
for k in range(1,9):
 a=np.sqrt(k*(k+1)); A[k,:k]=1/a;A[k,k]=-k/a

def block(x):return x.reshape(28,9,28,9).transpose(0,2,1,3)
def unblock(x):return x.transpose(0,2,1,3).reshape(SIDE,SIDE)
def f2(x,M):return np.einsum('ia,...ab,jb->...ij',M,x,M,optimize=True)
def i2(x,M):return np.einsum('ia,...ij,jb->...ab',M,x,M,optimize=True)
def haar_1d(v,inv=False):
 n=v.shape[-1];h=n//2
 if not inv:
  return np.concatenate(((v[...,:n:2]+v[...,1:n:2])/np.sqrt(2),(v[...,:n:2]-v[...,1:n:2])/np.sqrt(2)),axis=-1)
 z=np.empty_like(v);z[...,::2]=(v[...,:h]+v[...,h:])/np.sqrt(2);z[...,1::2]=(v[...,:h]-v[...,h:])/np.sqrt(2);return z

def haar2(x,inverse=False):
 y=x.copy()
 if not inverse:
  for size in (SIDE,SIDE//2):
   q=y[:size,:size]; y[:size,:size]=haar_1d(haar_1d(q).T).T
 else:
  for size in (SIDE//2,SIDE):
   q=y[:size,:size]; y[:size,:size]=haar_1d(haar_1d(q.T,True).T,True)
 return y
D=dct(np.eye(9),type=2,norm='ortho',axis=0)
assert np.max(np.abs(A@A.T-np.eye(9)))<1e-14
methods=['H9T','DCT-II','Haar-L2']
def enc(x,m):return haar2(x) if m=='Haar-L2' else f2(block(x),A if m=='H9T' else D)
def dec(z,m):return haar2(z,True) if m=='Haar-L2' else unblock(i2(z,A if m=='H9T' else D))
def codec(y,rate):
 flat=y.ravel(); k=max(1,int(rate*len(flat)));idx=np.argpartition(np.abs(flat),-k)[-k:]
 mask=np.zeros(len(flat),dtype=np.uint8);mask[idx]=1
 # Fixed uniform quantization, conservative shared quantization step for all transforms
 q=np.clip(np.rint(flat[idx]/0.002),-32768,32767).astype('<i2')
 zz=np.zeros(len(flat));zz[idx]=q.astype(float)*0.002
 payload=np.packbits(mask).tobytes()+q[np.argsort(idx)].tobytes()
 return zz.reshape(y.shape), len(zlib.compress(payload,level=9))
rows=[]
imgs={'camera':data.camera().astype(float)/255,'coins':data.coins().astype(float)/255,'astronaut':color.rgb2gray(data.astronaut())}
for name,img in imgs.items():
 x=img[:SIDE,:SIDE]
 for m in methods:
  y=enc(x,m);print('ROUNDTRIP',name,m,np.max(np.abs(dec(y,m)-x)));assert np.max(np.abs(dec(y,m)-x))<2e-13
  for rate in (.05,.10,.20):
   z,nb=codec(y,rate);r=dec(z,m);mse=np.mean((r-x)**2)
   rows.append(dict(task='image_rate_distortion',dataset=name,seed='',method=m,parameter=rate,psnr_db=10*np.log10(1/mse),bpp=8*nb/x.size,auc='',ap=''))
# Industrial-like anomaly detection, synthetic independently randomized windows, disjoint train/test.
# Three types: impulse, short burst, drift. Features from 9 point block coefficients; matched classifiers.
def feature(x,m):
 if m=='Raw':return x
 if m=='H9T':return x@A.T
 if m=='DCT-II':return x@D.T
 if m=='FFT-magnitude':return np.abs(np.fft.rfft(x,axis=1,norm='ortho'))
 if m=='Haar':
  # a reversible 8-sample Haar + unmodified 9th point, avoiding padding leakage
  v=x[:,:8].copy()
  for n in (8,4,2): v[:,:n]=haar_1d(v[:,:n])
  return np.column_stack((v,x[:,8]))
 raise ValueError(m)
def get_windows(seed,n=2400):
 rng=np.random.default_rng(seed)
 pos=rng.choice(n,n//2,replace=False);labels=np.zeros(n,dtype=int);labels[pos]=1
 t=np.arange(9)
 phase=rng.uniform(0,2*np.pi,(n,1)); freq=rng.uniform(.12,.32,(n,1))
 clean=.25*np.sin(2*np.pi*freq*t+phase)+rng.normal(0,.055,(n,9))
 x=clean.copy(); typ=rng.integers(0,3,n)
 for i in pos:
  if typ[i]==0:x[i,rng.integers(0,9)]+=rng.choice([-1,1])*.28
  elif typ[i]==1:
   k=rng.integers(0,7);x[i,k:k+3]+=rng.choice([-1,1])*.16
  else:x[i,:]+=(t-4)*rng.choice([-1,1])*.045
 return x,labels
for seed in (11,23,37):
 xtr,ytr=get_windows(seed,2400);xte,yte=get_windows(seed+1000,1600)
 for m in ['Raw','H9T','DCT-II','FFT-magnitude','Haar']:
  model=HistGradientBoostingClassifier(max_iter=65,max_leaf_nodes=11,learning_rate=.07,random_state=123,l2_regularization=1.0)
  tr=feature(xtr,m);te=feature(xte,m); model.fit(tr,ytr);p=model.predict_proba(te)[:,1]
  rows.append(dict(task='anomaly_heldout',dataset='mixed_spike_burst_drift',seed=seed,method=m,parameter='',psnr_db='',bpp='',auc=roc_auc_score(yte,p),ap=average_precision_score(yte,p)))
with open(OUT/'followup_results.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
for row in rows:
 if row['task']=='image_rate_distortion' and row['parameter']==.1:print('IMAGE',row['dataset'],row['method'],round(row['psnr_db'],2),'PSNR',round(row['bpp'],3),'bpp')
 if row['task']=='anomaly_heldout':print('ANOMALY',row['seed'],row['method'],round(row['auc'],4),round(row['ap'],4))
