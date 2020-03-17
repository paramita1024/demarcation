import getopt
from sklearn.linear_model import Ridge, HuberRegressor, Lasso
import datetime
import time 
import numpy  as np
from numpy import linalg as LA
# from slant import slant
from myutil import * 
# from data_preprocess import *
import sys
# import Scipy
from numpy import linalg as LA


class Huber_loss_minimization:


	def __init__( self, data, set_of_alpha, set_of_epsilon ):
		self.data = data
		self.set_of_alpha= set_of_alpha
		self.set_of_epsilon = set_of_epsilon
		self.generate_subset()
		
	def generate_subset( self):
		
		res={}
		for alpha in self.set_of_alpha:
			res[ str( alpha)] = {}
			for epsilon in self.set_of_epsilon:


				self.w={}
				self.o = { str(user) : \
					np.zeros( self.data['per_user'][str( user ) ]['A'].shape[0] ) for user in self.data['nodes'] }
				huber = HuberRegressor( epsilon = epsilon, alpha = alpha )			
				for user in self.data['nodes']:
					user_data = self.data['per_user'][str( user ) ]
					if user_data['A'].shape[0] > 0 : 
						huber.fit( user_data['A'], user_data['b']  )
						w_user = huber.coef_
						b_user = huber.intercept_
						self.w[ str(user) ] = {'w':w_user, 'b':b_user }
						residual = user_data['b'] - huber.predict( user_data['A'] )
						self.o[ str(user) ] =  residual * huber.outliers_

				num_msg = self.data['all_user']['train'].shape[0]
				outlier = np.zeros( num_msg  )
				for user in self.data['nodes']:
					outlier[ self.data['msg_index'][ str( user )] ] = self.o[ str(user) ]
				res[str( alpha )][ str( epsilon )] = { 'indices' : np.argsort( outlier ) ,'w' : self.w, 'outlier' : outlier }
		self.res=res

class Extended_robust_lasso:


	def __init__( self, data, set_of_lamb_w, set_of_lamb_e ):

		self.data = data

		self.epsilon = 0.0001
		self.max_iter = 500 # 00 # check
		
		self.set_of_lamb_w= set_of_lamb_w
		self.set_of_lamb_e = set_of_lamb_e
		
		self.generate_subset()
		
	def generate_subset( self):

		def soft_threshold( r , l ):
			tmp = np.absolute(r) - l
			tmp[ tmp < 0 ] = 0 
			return np.sign( r )*tmp

		res={}
		
		for lamb_w in self.set_of_lamb_w:
			res[ str( lamb_w)]={}
			lasso = Lasso( alpha = lamb_w )
			for lamb_e in self.set_of_lamb_e:
				self.w={}
				for user in self.data['nodes']:
					if self.data['per_user'][str( user ) ]['A'].shape[0] > 0 :
						self.w[ str( user )] = {'w':np.zeros( self.data['per_user'][str( user ) ]['A'].shape[1] ), 'b':0}
				
				self.o = { str(user) : np.zeros( self.data['per_user'][str( user ) ]['A'].shape[0] )\
					 for user in self.data['nodes'] }
				
				itr=0
				list_of_del_w = []
				start = time.time()
				while True:
					del_w = 0 
					for user in self.data['nodes']:#*
						user_data = self.data['per_user'][str( user ) ]
						if user_data['A'].shape[0]>0:
							lasso.fit( user_data['A'], user_data['b'] - self.o[ str( user )] )
							
							w_user = lasso.coef_
							b_user = lasso.intercept_
							del_w += LA.norm( self.w[ str(user) ]['w'] - w_user ) + \
								np.abs( self.w[ str(user) ]['b'] - b_user )
							self.w[ str(user) ] = {'w':w_user, 'b':b_user }
							residual = user_data['b'] - lasso.predict( user_data['A'] )
							self.o[ str(user) ] = soft_threshold( residual , lamb_e )

					end = time.time()
					# print 'Iteration ', itr, ' takes ', end-start, ' seconds '
					start = end 
					list_of_del_w.append( del_w )
					if del_w  < self.epsilon :
						break

					itr+=1
					if itr == self.max_iter:
						break

				num_msg = self.data['all_user']['train'].shape[0]
				outlier = np.zeros( num_msg	 )
				for user in self.data['nodes']:
					outlier[ self.data['msg_index'][ str( user )] ] = self.o[ str(user) ]
				res[str( lamb_w )][ str( lamb_e )] = { 'indices' : np.argsort( outlier ),\
											'w' : self.w, 'outlier' : outlier, 'list_of_del_w': list_of_del_w }

		self.res=res

class Soft_thresholding:

	def __init__( self, data,  set_of_lamb_w, set_of_lamb_e ):
		
		self.data = data
		# for user in self.data['nodes']:
		# 	# print user
		# 	if len(self.data['per_user'][ str( user )]['A'].shape)<2:
		# 		print '*'*50,'yes'
		# 		print self.data['msg_index'][ str(user)].shape
		# 		print self.data['per_user'][ str( user )]['A'].shape[0]
			# print self.data['per_user'][ str( user )]['b'].shape

		self.set_of_lamb_w= set_of_lamb_w
		self.set_of_lamb_e = set_of_lamb_e
		
		self.epsilon = 0.0001
		self.max_iter = 500 # 000 # check 
		self.generate_subset()

		
	def generate_subset( self):

		def soft_threshold( r , l ):
			tmp = np.absolute(r) - l
			tmp[ tmp < 0 ] = 0 
			return np.sign( r )*tmp

		res={}
		self.w={}
		for user in self.data['nodes']:
			if self.data['per_user'][str( user ) ]['A'].shape[0] > 0 :
				self.w[ str( user )] = {'w':np.zeros( self.data['per_user'][str( user ) ]['A'].shape[1] ), 'b':0}
		self.o = { str(user) : np.zeros( self.data['per_user'][str( user ) ]['A'].shape[0] ) for user in self.data['nodes'] }
		for lamb_w in self.set_of_lamb_w:
			res[str(lamb_w)]={}
			rr = Ridge( alpha = lamb_w )
			for lamb_e in self.set_of_lamb_e:
				# print self.data['nodes']
				
				itr=0
				# start = time.time()
				list_of_del_w = []
				while True:
					# print itr
					del_w = 0 
					for user in self.data['nodes']:

						user_data = self.data['per_user'][str( user ) ]
						if user_data['A'].shape[0] > 0 :
							rr.fit( user_data['A'], user_data['b'] - self.o[ str( user )] )
							
							w_user = rr.coef_
							b_user = rr.intercept_
							del_w += LA.norm( self.w[ str(user) ]['w'] - w_user ) + \
								np.abs( self.w[ str(user) ]['b'] - b_user )
							self.w[ str(user) ] = {'w':w_user, 'b':b_user }
							residual = user_data['b'] - rr.predict( user_data['A'] )
							self.o[ str(user) ] = soft_threshold( residual.flatten() , lamb_e )

					if del_w  < self.epsilon :
						break
					list_of_del_w.append( del_w )
					itr += 1 
					# print time.time() - start , 'seconds'
					if itr == self.max_iter:
						break

				num_msg = self.data['all_user']['train'].shape[0]
				outlier = np.zeros( num_msg  )
				for user in self.data['nodes']:
					outlier[ self.data['msg_index'][ str( user )] ] = self.o[ str(user) ]
				res[str( lamb_w )][ str( lamb_e )] = { 'indices' : np.argsort( outlier ),\
											'w' : self.w , 'outlier' : outlier, 'list_of_del_w': list_of_del_w }

		self.res=res
