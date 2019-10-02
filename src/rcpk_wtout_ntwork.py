import datetime
import time 
import numpy  as np
from numpy import linalg as LA
# from slant import slant
from myutil import * 
from data_preprocess import *
import sys
class softRCPK_wtout_network:
	def __init__(self,obj,param):

		self.train = obj.train
		self.nodes = obj.nodes
		self.test = obj.test
		self.num_node= self.nodes.shape[ 0 ]
		self.num_train= self.train.shape[ 0 ]
		self.num_test = self.test.shape[ 0 ]
		self.msg_window=param['win']
		self.lamb =param['lamb']
		self.frac=param['frac']

	def create_opinion_matrix(self):
		self.msg=np.concatenate((self.train[:,2],self.test[:,2]),axis=0) 
		self.opinion_matrix=np.zeros((self.num_train,self.msg_window))
		for ind in range(1,self.num_train):
			end_ind=max(ind-self.msg_window,0)
			self.opinion_matrix[ind, :min([ind,self.msg_window])]=np.flipud(self.msg[end_ind:ind] )
			
	def init_DS( self ):
		self.create_opinion_matrix()
		self.msg_user_index=np.zeros((self.num_train,self.num_node))
		for ind in range(self.num_train): 
			self.msg_user_index[ind,int(self.train[ind,0])] = 1
	
	def RR_via_ST(self,max_itr):
		itr = 0 
		w = np.zeros( self.msg_window + self.num_node )
		X=np.concatenate((self.msg_user_index, self.opinion_matrix), axis=1)
		Y=self.train[:,2]
		weights=np.ones(self.num_train)*(1/self.num_train)
		while itr < max_itr:
			w= self.update(X,Y,self.lamb,weights)
			residual=self.get_residual( Y, X, w)
			weights=self.set_weights(residual)
			itr += 1 
		return self.hard_threshold( residual,self.frac)
	
	def get_residual(self, Y, X, w ):
		return np.absolute( Y - X.dot( w ) )

	def update(self,X,Y,lamb,prob):
		model = Ridge(alpha = lamb,fit_intercept=False).fit(X, Y, sample_weight = prob)
		return model.coef_

	def hard_threshold( self , v , frac  ):
		num_smallest_val = int( frac * v.shape[ 0 ] )
		idx =np.argpartition( v , num_smallest_val-1 )[:num_smallest_val]  # ck 
		active_set = np.zeros( self.num_train, dtype=bool)
		active_set[ idx ] = True
		return active_set

	def set_weights(self,residue):
		residue=residue/float(np.sum(residue))
		return (1-residue)
