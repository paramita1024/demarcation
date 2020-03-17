import scipy.linalg as SCLA
import getopt 
import time
#import matplotlib.pyplot as plt
import numpy as np
from slant import *
from myutil import *
import math


def parse_command_line_input( list_of_file_name ):

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, 'l:f:', ['lamb','file_name'])

    lamb=0.5
    file_name=''
    
    for opt, arg in opts:
        if opt == '-l':
            lamb = float(arg)
        if opt == '-f':
            for file_name_i in list_of_file_name:
            	if file_name_i.startswith( arg ):
            		file_name = file_name_i
    return file_name, lamb 

class ecpk:
	
	def __init__( self , obj = None, sigma_covariance = 1., lamb = 1.0, w=10.0, batch_size=1 ):
		

		self.train = obj['train']
		self.test = obj['test']
		self.edges = obj['edges']	

		self.num_node= self.edges.shape[ 0 ]
		self.num_train= self.train.shape[ 0 ]
		self.num_test = self.test.shape[ 0 ]
		self.nodes = np.arange( self.num_node )

		self.sigma_covariance = sigma_covariance
		self.lamb = lamb
		self.w = w
		self.batch_size=batch_size
		self.threshold = 1e-3
		self.max_iter = 10000
	
	def create_influence_matrix(self):
		influence_matrix = np.zeros(( self.num_train, 1+self.num_node)) 
		influence_matrix[:,0] = 1 
		msg_index = 0
		time_old = 0

		reminder = {}
		for user, time, sentiment in self.train : 
			user = int(user)
			if msg_index > 0 :
			    influence_matrix[msg_index, 1:] = influence_matrix[msg_index-1 , 1:]*np.exp(-self.w*(time - time_old) ) 
			    influence_matrix[msg_index, reminder['user']+1] += reminder['sentiment']*np.exp(-self.w*(time - time_old)) 
			reminder['user'] = user 
			reminder['sentiment'] = sentiment
			msg_index += 1
			time_old = time
		self.influence_matrix = influence_matrix
		return influence_matrix
	
	def set_c( self ): # ck 
		max_msg_influ_mat = np.max( np.absolute( self.influence_matrix ) , axis = 1 ) ** 2 
		tmp = np.zeros( self.num_node )
		for user in self.nodes:
			msg_idx = np.where( self.train[:,0] == user )[0]
			tmp[ user ] = np.sum( max_msg_influ_mat[ msg_idx ]  )
		# print 'check c of cherrypick'
		# print np.max(tmp)
	
		self.lamb =  5*np.max(tmp)/(self.sigma_covariance**2)
	
	def create_neighbours(self):
		self.incremented_nbr={}

		self.msg_per_user={}
		for user in self.nodes:
			neighbours = np.nonzero(self.edges[:,user].flatten())[0]
			self.incremented_nbr[user]=np.concatenate((np.array([0]),neighbours+1))
			self.msg_per_user[user] = np.array([ index for index in range( self.num_train ) \
				if int(self.train[index][0]) == user ])
	
	def create_covariance_matrix(self):
		self.covar={}
		for user in self.nodes:
			self.covar[user]= self.lamb * np.eye(self.incremented_nbr[user].shape[0])

	def create_min_eig_dictionary( self ):

		self.min_eig={}
		self.min_eig['users'] = self.lamb * np.ones( self.num_node )
		self.min_eig['others_global_min'] = self.lamb * np.ones( self.num_node )
		self.min_eig['global']=np.amin( self.min_eig['users'])
		self.min_eig['msg'] = np.zeros( self.num_train )
		self.inc = np.zeros( self.num_train )
		for user in self.nodes:
			for m in self.msg_per_user[user]:
				msg_vector = self.get_influence_vector( user, m )
				msg_vector = msg_vector.reshape( msg_vector.shape[0], 1)
				covar_inc = self.covar[user] + ( 1/self.sigma_covariance**2 ) * msg_vector.dot(msg_vector.T)
				min_eig = self.min_eig_power( covar_inc )
				self.min_eig['msg'][ m ] = min_eig
				if min_eig >= self.lamb : 
					self.inc[ m ] = 0
				else:
					self.inc[ m ] = min_eig - self.lamb
	
	def create_init_data_structures( self ):
		start = time.time()
		self.create_neighbours()
		self.create_influence_matrix() 
		self.create_covariance_matrix()
		self.create_min_eig_dictionary()
		now=time.time()
		# print (now - start), ' seconds '
		# return 
		self.msg_end = np.zeros( self.num_train, dtype = bool )
		self.list_of_msg=[]

	def get_influence_vector(self,user, msg_num):
		return self.influence_matrix[msg_num][self.incremented_nbr[user]].flatten()
	
	def min_eig_power(self, X ):
		
		#
		w = SCLA.eigh(X, eigvals_only=True)
		# print w.shape[0]
		# print w
		return w[0]	
		#
		# max_eig = self.dom_eig_power( X )
		# max_eig=1#
		# X -= max_eig* np.eye( X.shape[0])
		# min_eig = self.dom_eig_power( X ) + max_eig 
		# return min_eig

	# def dom_eig_power(self, A):

	# 	v = np.ones( (A.shape[0],1) )
	# 	v /= LA.norm(v)
	# 	Av = A.dot( v )
	# 	for iter in range( self.max_iter):
	# 		v = Av / LA.norm(Av)
	# 		Av=A.dot(v)
	# 		lamb = v.T.dot(Av)
	# 		if LA.norm( Av-lamb*v) < self.threshold:
	# 			return lamb
	# 	# return float('inf')
			 

		
	def update( self, msg, user ):
		
		v = self.get_influence_vector( user, msg )
		v = v.reshape( v.shape[0] , 1)
		self.covar[user] += ( 1/self.sigma_covariance**2 )*( v.dot(v.T) )
		self.min_eig['users'][ user ]  = self.min_eig['msg'][msg]
		self.min_eig['global'] = np.amin( self.min_eig[ 'users' ] ) 
		for u in self.nodes:
			if self.min_eig['users'][u] > self.min_eig['global']:
				self.min_eig['others_global_min'][u] = self.min_eig['global']
			else:
				self.min_eig['others_global_min'][u] = min( [ \
					self.min_eig['users'][other_u] for other_u in self.nodes if other_u != u  ])

		
		for u in self.nodes:
			if u == user: 
				for m in self.msg_per_user[user]: 
					if not self.msg_end[m]:
						msg_vector = self.get_influence_vector( user, m )
						msg_vector = msg_vector.reshape( msg_vector.shape[0], 1)
						covar_inc = self.covar[user] + ( 1/self.sigma_covariance**2 ) * msg_vector.dot(msg_vector.T)
						min_eig = self.min_eig_power( covar_inc )
						if min_eig > self.min_eig['others_global_min'][user] :
							self.inc[ m ] = 0
						else:
							self.inc[ m ] = min_eig - self.min_eig['global']
						self.min_eig['msg'][ m ] = min_eig
			else:
				for m in self.msg_per_user[u] :
					if not self.msg_end[m]:
						if self.min_eig['msg'][m ] >= self.min_eig['others_global_min'][u] :
							self.inc[m] = 0 
						else:
							self.inc[m] = self.min_eig['msg'][m] - self.min_eig['global']
				

	def obtain_most_endogenius_msg_user(self):
		msg_to_choose = np.argmax( self.inc )  
		self.inc[msg_to_choose] = - float('inf')
		corr_user  = int( self.train[ msg_to_choose , 0 ] )
		
		if self.msg_end[msg_to_choose]:
			print( " A message which is already endogenious has been selected again as endogenious msg")
		
		self.msg_end[ msg_to_choose ] = True
		self.list_of_msg.append( msg_to_choose)
		self.update( msg_to_choose , corr_user )
		# if max(inc) == -float('inf'): 
		# 	print( "The maximum entry in increment array is - Infinity ")


	def demarkate_process(self, res_file ): 

		self.create_init_data_structures() 
		# return 
		num_end_msg = 0 
		start=time.time()
		while num_end_msg < self.num_train :
                    self.obtain_most_endogenius_msg_user()
                    end=time.time()
                    num_end_msg += 1 
		res={}
		res['data'] = np.array( self.list_of_msg )
		res['w']=self.w
		res['sigma_covariance'] = self.sigma_covariance
                save(res, res_file)
def main():

	list_of_file_name = ['barca','british_election','GTwitter',\
	'jaya_verdict', 'JuvTwitter' , 'MsmallTwitter',\
	'Twitter' , 'VTwitter']
        file_name,lamb = parse_command_line_input( list_of_file_name )
	list_of_lambda = [0.01,0.05,0.1,0.2,0.3,0.4]#[.5,.7,1.,1.5,2.]

	w=load_data('w_v')[file_name]['w']
	v=load_data('w_v')[file_name]['v']

	data_file = '../Data/' + file_name 
	data_all = load_data(data_file)
        #print(data_all['all_user'].keys())
        #eturn
        data = {'nodes': data_all['nodes'], 'edges': data_all['edges'] , 'train': data_all['all_user']['train'] , \
            'test': data_all['all_user']['test']  }
	res_file = '../Result_Subset/' + file_name
	
	for lamb in list_of_lambda:
		slant_obj = slant( obj= data , init_by = 'dict'  , data_type = 'real', tuning = True, tuning_param = [w,v,lamb] ) 
		sigma = slant_obj.get_sigma_covar_only()
		del slant_obj
		
		start = time.time()
		obj = ecpk( obj = data, sigma_covariance = sigma, lamb =  lamb, w=w ) 
		
		res_file_l  = res_file + '.l' + str(lamb) + '.ecpk' 
		obj.demarkate_process( res_file_l )
		total_time = time.time() - start
	
		del obj
	
	print(file_name + ' done ')




	
if __name__== "__main__":
  main()


