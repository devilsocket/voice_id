import os
import pickle
import numpy as np 
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM 
from sklearn import preprocessing
import python_speech_features as mfcc
from pprint import pprint
from tqdm import tqdm
#-------------------------------------------------------------------------------------------
logo = """
██╗   ██╗ ██████╗ ██╗ ██████╗███████╗    ██╗██████╗ 
██║   ██║██╔═══██╗██║██╔════╝██╔════╝    ██║██╔══██╗
██║   ██║██║   ██║██║██║     █████╗      ██║██║  ██║
╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝      ██║██║  ██║
 ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗    ██║██████╔╝
  ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝    ╚═╝╚═════╝ 
	(TRAINER) VERSION 1.0                                                     
"""
#-------------------------------------------------------------------------------------------
def calculate_delta(array):
	"""Calculate and returns the delta of given feature vector matrix"""
	rows,cols = array.shape
	deltas = np.zeros((rows,20))
	N = 2
	for i in range(rows):
		index = []
		j = 1
		while j <= N:
			if i-j < 0:
				first = 0
			else:
				first = i-j
			if i+j > rows -1:
				second = rows -1
			else:
				second = i+j
			index.append((second,first))
			j+=1
		deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
	return deltas

def extract_features(audio,rate):
	"""extract 20 dim mfcc features from an audio, performs CMS and combines 
	delta to make it 40 dim feature vector"""    
	mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01,20,appendEnergy = True)
	
	mfcc_feat = preprocessing.scale(mfcc_feat)
	delta = calculate_delta(mfcc_feat)
	combined = np.hstack((mfcc_feat,delta)) 
	return combined
#------------------------------------------------------------------------------------------------------------------------


curr_dir = os.getcwd()
train_dataset_path = os.path.join(curr_dir,'train')
models_path = os.path.join(curr_dir,'models')

def execute():
	identities = {}
	for root, dirs, files in os.walk(train_dataset_path):
		for file in files:
			if file.endswith('.wav'):
				fpath = os.path.join(root,file)
				identity = fpath.split(os.sep)[-3].split('-')[0]
				if identity not in identities:identities[identity]=[]
				else:identities[identity].append(fpath)
	if identities:
		for identity in identities:
			features = np.asarray(())
			for sample in tqdm(identities[identity],desc=identity,ncols=100):
				sr,audio = read(sample)
				vector   = extract_features(audio,sr)
				if features.size == 0:features = vector
				else:features = np.vstack((features, vector))
			gmm = GMM(n_components = 16, max_iter = 200, covariance_type='diag',n_init = 3)
			gmm.fit(features)
			model_file_path = os.path.join(models_path,'{}.gmm'.format(identity))
			pickle.dump(gmm,open(model_file_path,'wb'))
			print('[+] model saved for speaker {} with shape {}'.format(identity,features.shape))
	return 1

if __name__ == '__main__':
	print(logo)
	execute()