import numpy  as np
from myutil import * 
from data_preprocess import *
from sklearn.linear_model import Ridge

class softRCPK:
	def __init__(self,obj,param):
		self.train = obj.train
		self.edges = obj.edges
		self.test = obj.test
		
		self.num_node= self.edges.shape[ 0 ]
		self.num_train= self.train.shape[ 0 ]
		self.num_test = self.test.shape[ 0 ]
		self.nodes = np.arange( self.num_node )
		
		self.lamb = param['lamb']
		self.w_slant=param['w']
		self.frac=param['frac']

	def init_DS( self ):
		
		self.msg_index={}
		self.msg_mask={}
		self.nbr = {}
		
		for user in self.nodes:
			self.msg_index[ user ]=  np.where( self.train[:,0] == user )[0] 
			self.msg_mask[ user ] =np.ones( self.msg_index[ user ].shape[ 0 ] , dtype=bool)
			self.nbr[ user ] = np.nonzero( self.edges[:,user].flatten() )[0] 
				
		self.influence_matrix = {}
		for user in self.nodes : 
			if self.msg_index[ user ].shape[0] == 0 :
				influence_matrix_user = np.array([])
			else:
				influence_matrix_user = np.zeros(( self.msg_index[user].shape[0], 1+ self.nbr[user].shape[0]))
				nbr_idx = 0 
				influence_matrix_user[:,0] = 1
				for his_nbr in self.nbr[user]:
					if self.msg_index[ his_nbr ].shape[0] > 0 :
						time_old = 0 
						m_idx = 0
						opn = 0 
						merged_index = np.sort( np.concatenate( ( self.msg_index[ user ], self.msg_index[ his_nbr ])))
						for index in merged_index:
							user_curr, time_curr , sentiment = self.train[ index,: ]
							opn *= np.exp( - self.w_slant*( time_curr - time_old)) 
							if user_curr == user:
								influence_matrix_user[ m_idx , 1 + nbr_idx] = opn 
								m_idx += 1  
							else:
								opn += sentiment
							time_old = time_curr
					nbr_idx += 1
			self.influence_matrix[user] = influence_matrix_user  

	def RR_via_soft_threshold( self,max_itr ):
		itr = 0 
		residual_all_user = np.zeros(self.num_train)
		weights=np.ones(self.num_train)*(1/self.num_train)
		while itr < max_itr:
			print itr
			for user in self.nodes:
				if self.msg_index[user].shape[0] > 0 :
					X=self.influence_matrix[ user ]
					Y=self.train[ self.msg_index[ user ],2]
					w_user= self.update( X,Y,self.lamb,weights[self.msg_index[user]])
					residual_all_user[self.msg_index[ user ]] =self.get_residual(Y,X,w_user)
			weights=self.set_weights(residual_all_user)
			itr += 1 
		return self.hard_threshold( residual_all_user , self.frac )

	def set_weights(self,residue):
		residue=residue/float(np.sum(residue))
		return (1-residue)

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
	