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
	(EVALUATE) VERSION 1.0                                                     
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
test_dataset_path = os.path.join(curr_dir,'test')
models_path = os.path.join(curr_dir,'models')
models = []
identities = {}
id_count=0
for root, dirs, files in os.walk(models_path):
	for file in files:
		if file.endswith('.gmm'):
			identity=file.replace('.gmm','')
			identities[id_count]=identity
			mpath=os.path.join(root,file)
			models.append(pickle.load(open(mpath,'rb')))
			id_count+=1

def execute():
	tests = []
	for root, dirs, files in os.walk(test_dataset_path):
		for file in files:
			if file.endswith('.wav'):
				fpath = os.path.join(root,file)
				tests.append(fpath)
	if tests:
		for test in tests:
			sr,audio=read(test)
			vector=extract_features(audio,sr)
			predict_probability = np.zeros(len(models))
			for i in range(len(models)):
				gmm=models[i]
				scores=np.array(gmm.score(vector))
				predict_probability[i] = scores.sum()
			prediction = np.argmax(predict_probability)
			subject_name = identities[prediction]
			print('[+] audio: {} >> id: {}'.format(test,subject_name))
	return 1

if __name__ == '__main__':
	print(logo)
	execute()