import getopt 
import time
#import matplotlib.pyplot as plt
import numpy as np
from slant import *
from myutil import *


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

class acpk:
	
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
		for user in self.nodes:
			neighbours = np.nonzero(self.edges[:,user].flatten())[0]
			self.incremented_nbr[user]=np.concatenate((np.array([0]),neighbours+1))
	
	def create_covariance_inverse(self):
		self.covariance_inverse={}
		for user in self.nodes:
			self.covariance_inverse[user]=np.eye(self.incremented_nbr[user].shape[0])/self.lamb
	
	def create_init_data_structures( self ):
		self.create_neighbours()
		self.create_influence_matrix() 
		self.create_covariance_inverse()
		self.msg_end = np.zeros(self.num_train, dtype = bool)
		self.list_of_msg=[]
	
	def get_influence_vector(self,user, msg_num):
		return self.influence_matrix[msg_num][self.incremented_nbr[user]].flatten()
	
	def obtain_most_endogenius_msg_user(self):

		inc = - float('inf') * np.ones( self.num_train) 
		for msg_no in np.nonzero(~self.msg_end)[0]: 
			user = int(self.train[msg_no,0]) 
			x = self.get_influence_vector(user, msg_no)
			A_inv_x = self.covariance_inverse[user].dot( x ).flatten()
			w_T_w =  A_inv_x.dot( A_inv_x )
			inc[msg_no] = - w_T_w / ( self.sigma_covariance**2 + A_inv_x.dot( x )  )
			
		for itr in range(self.batch_size):
			msg_to_choose = np.argmax(inc) 
			inc[msg_to_choose] = - float('inf')
			corr_user  = int( self.train[ msg_to_choose , 0 ] )
			
			if self.msg_end[msg_to_choose]:
				print( " A message which is already endogenious has been selected again as endogenious msg")
			
			self.msg_end[ msg_to_choose ] = True
			self.list_of_msg.append( msg_to_choose)
			self.update_inverse( msg_to_choose , corr_user )
			if max(inc) == -float('inf'): 
				print( "The maximum entry in increment array is - Infinity ")
	
	def update_inverse(self, msg_num, user ):
		influence_vector=self.get_influence_vector(user,msg_num)
		w = (self.covariance_inverse[user].dot( influence_vector)).reshape(influence_vector.shape[0],1)
		self.covariance_inverse[user] -= (w.dot(w.T))/( self.sigma_covariance**2 + influence_vector.dot(w))
	
	def demarkate_process(self, res_file ): 

		self.create_init_data_structures() 
		num_end_msg = 0 
		start=time.time()
		while num_end_msg < self.num_train :
                    self.obtain_most_endogenius_msg_user()
                    end=time.time()
                    num_end_msg += 1 # self.batch_size	
		res={}
		res['data'] = np.array( self.list_of_msg )
		res['w']=self.w
		res['sigma_covariance'] = self.sigma_covariance
                save(res, res_file)
def main():

	list_of_file_name = ['barca','british_election','GTwitter',\
	'jaya_verdict', 'JuvTwitter' , 'MlargeTwitter', \
	'MsmallTwitter', 'real_vs_ju_703', 'trump_data', 'Twitter' , 'VTwitter']
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
		cherrypick_obj = acpk( obj = data, sigma_covariance = sigma, lamb =  lamb, w=w ) 
		
		res_file_l  = res_file + '.l' + str(lamb) + '.acpk' 
		cherrypick_obj.demarkate_process( res_file_l )
		total_time = time.time() - start
	
		del cherrypick_obj
	
	print(file_name + ' done ')




	
if __name__== "__main__":
  main()


