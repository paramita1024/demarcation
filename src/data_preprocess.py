import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
# from myutil import *
# from slant import results


def load_data(input_file,flag=None):
	if flag=='ifexists':
		if not os.path.isfile(input_file+'.pkl'):
			# print 'not found', input_file
			return {}
	# print 'found'
	with open(input_file+'.pkl','rb') as f:
		data = pickle.load(f)
	return data
		
def save_data(obj, write_file):
	# data_obj = data(graph , train, test )
	with open(write_file+'.pkl','wb') as f:
		pickle.dump( obj, f, pickle.HIGHEST_PROTOCOL)
def save(obj, write_file):
	# data_obj = data(graph , train, test )
	with open(write_file+'.pkl','wb') as f:
		pickle.dump( obj, f, pickle.HIGHEST_PROTOCOL)
	
# done ----------------------------------
# Twitter_10ALLXContainedOpinionX
# VTwitter_10ALLXContainedOpinionX
# trump _ data 
# real vs ju 703
# MsmallTwitter
# M large
# Juv
# jaya verdict

def generate_dataset_info( directory, file_ext ):
	# directory = '../Cherrypick_others/Data_opn_dyn_python/'
	list_of_files = os.listdir( directory )
	file_info = []
	for file in list_of_files :
		if file.endswith( file_ext ):
			obj = load_data( directory + file[:-4] )
			# opinion = np.concatenate((obj.train[:,2], obj.test[:,2]), axis=0)
			# plt.plot(opinion)
			# plt.show()
			# plt.clf()
			# print file
			dataset_title = file.split('_10ALLX')[0]
			
			file_description = {}
			file_description['name'] = dataset_title.replace( '_',' ')
			file_description['n_sample'] = obj.train.shape[0] + obj.test.shape[0]
			file_description['n_nodes'] = obj.edges.shape[0]
			file_description['n_edges'] = np.count_nonzero(obj.edges)/2
			opinion = np.concatenate((obj.train[:,2], obj.test[:,2]), axis=0)
			file_description['mean_opn']=np.mean(opinion)
			file_description['std_opn']=np.std(opinion)
			file_info.append( file_description )
	# print '\\begin{center}\n\\begin{tabular}{|c|c|c|}\n\\hline'
	# print 'Dataset & No of sample & No of Nodes \\\\'

	# print '\\hline \n'
	# return
	for file in file_info :
		str_curr=file['name']
		for key in ['n_nodes','n_edges','n_sample','mean_opn','std_opn']:
			if key in ['mean_opn','std_opn']:
				str_curr+=' & '+str(file[key])[:5]
			else:
				str_curr+=' & '+str(file[key])

		print str_curr,'\\\\'
	# print '\\hline\n\\end{tabular}\n\\end{center}'

def plot_opinion( file_prefix ):
	f=plt.figure()
	file_to_save = '../Plots/Opinion_Histogram/' + file_prefix + '.opinion_histogram.png'
	obj = load_data( '../Cherrypick_others/Data_opn_dyn_python/' + file_prefix + '_10ALLXContainedOpinionX.obj')
	opn = np.concatenate(( obj.train[:,2] , obj.test[:,2] ), axis = 0)
	N_bin = 20
	plt.hist( opn, bins = N_bin )
	plt.title(file_prefix)
	# plt.show()
	plt.savefig( file_to_save )
	plt.clf()
	# print for kile
	# print '\\begin{figure}'
	# print '\\includegraphics[width=\\linewidth]{' +'../' + file_to_save + '}'   
  
	# print '\\end{figure}'



class data_preprocess:
	def __init__(self,file=None ):
		self.file = file
		# self.edges = edges
		# self.train = train
		# self.test = test
		# self.nuser = graph.get_nuser() # to be defined

	def generate_from_dict(self,data_dict):
		self.train=data_dict['train']
		self.test =data_dict['test']
		self.edges =data_dict['edges']


	def read_data(self):
		# read the graph
		# f = open ( 'input.txt' , 'r')
		# l = [[int(num) for num in line.split(',')] for line in f ]
		# print l
		# with open(self.file + '.metadata', 'r') as f:
		# 	for line in f :
		# 		metadata =  [ int(num) for num in line.split(',') ] 
		# 	# metadata is a list of int , 
		# 	# metadata = [ num_node ]
		# 	self.num_node = metadata[0]
		# 	self.nodes = range( self.num_node )
		# print self.num_node
		# print self.nodes

		with open(self.file + '.graph', 'r') as f:
			self.edges = np.array([ [ int(num) for num in line.split(',') ] for line in f ])
		self.num_node = self.edges.shape[0]
		# print self.num_node
		self.nodes = np.arange( self.num_node )
		# create adjacency matrix
		# print self.edges
		# # read the msg 
		with open(self.file + '.msg', 'r') as f:
			self.msg = np.array([ [ float(num) for num in line.split(',') ] for line in f ])
			self.num_msg  = self.msg.shape[0]
			print "num msg = "+ str(self.num_msg)
		self.msg = self.msg[np.argsort( self.msg[:,1] ), :]
		# plt.plot(self.msg[:,1])
		# plt.show()
        # print "number of user = "+str(self.num_node)
        # print "max user index= "+str(np.max(self.msg[:,0]))
        # print "min user index "+ str(np.min(self.msg[:,0]))
		# msg = msg[np.argsort( msg[:,1] ), :]

		# print self.msg
		
		# create 3 column array 

		# split the data 

		# create test and train 

		# save as python object 
	def split_data(self, train_fraction = 0 ):
		num_tr = int( self.num_msg * train_fraction )
		self.train = self.msg[:num_tr, :]
		self.test = self.msg[num_tr : ,:]
		print "num_train = " + str(self.train.shape[0])
		print "num_test = " + str(self.test.shape[0])
		del self.msg
		# print self.train

		# print "==========================="
		# print self.test


	def get_adj_graph_from_dict(self,file_name):
		def expand( v, list_of_nonzero):
			arr=np.zeros(v)
			for elm in list_of_nonzero:
				arr[elm-1]=1
			return arr 

		def check_symmetric(a, tol=1e-8):
			return np.allclose(a, a.T, atol=tol)

		file = open(file_name,'r') 
		graph=dict(zip(list(self.nodes+1),[[] for node in self.nodes]))
		for line in file: 
			start_node, dest_node=line.split(' ')
			# start_node=int(words[0]) 
			# dest_node=int(words[1])
			# if start_node in graph:
			graph[int(start_node)].append(int(dest_node))
			# else:
			# 	graph[start_node]=[dest_node]
		list_of_keys=np.array(graph.keys())
		# print'####################################'
		# print type(list_of_keys)
		print np.min(list_of_keys)
		print np.max(list_of_keys)
		# print list_of_keys.shape
		# return
		# print '###################################'

		# self.num_node=list_of_keys.shape[0]
		# print self.num_node
		# keys=graph.keys()
		# keys.sort()
		self.edges=np.array([expand(self.num_node, graph[start_node+1]) for start_node in self.nodes ])
		print self.edges.shape

		# print np.nonzero(self.edges[15])

		# print np.array(graph[16])-1

		# print check_symmetric(self.edges)

		# np.sum( self.edges, )


	

	def	map_edges(self, file_node, file_edge_old,file_edge_new):
		nodes=np.array([int(line) for line in open(file_node,'r')])
		print nodes.shape
		print np.unique(nodes).shape
		with open(file_edge_old,'r') as fr, open(file_edge_new,'w') as fw:
			for line in fr:
				original_nodes = map(int,line.split(' '))
				list_of_ind=[]
				for node in original_nodes:
					list_of_ind.append(np.where(nodes==node)[0][0]+1)
					

				fw.write(' '.join(map(str, list_of_ind ))+'\n')

	def	map_opinions(self, file_node, file_opinion_old,file_opinion_new):
		nodes=np.array([int(line) for line in open(file_node,'r')])
		with open(file_opinion_old,'r') as fr, open(file_opinion_new,'w') as fw:
			for line in fr:
				entries=line.split(' ')
				entries[0]=str(np.where(nodes==int(entries[0]))[0][0]+1)
				del entries[1]
				fw.write(' '.join(entries))
	def map_nodes(self, file_old,file_new ):
		with open(file_old,'r') as fr, open(file_new,'w') as fw:
			ind=1
			for line in fr:
				fw.write(str(ind)+'\n' )
				ind+=1

	def merge(self):
		self.train=np.concatenate((self.train,self.test),axis=0)
		self.test=[]

	def read_from_time_events(self, file_time, file_event,file_opinion,file_node):
		raw_data={'time':[], 'event':[]}

		with open( file_time, 'r') as f:
			ind=0
			for line in f:
				raw_data['time'].append(map(float, line.rstrip().split(' ')))
				ind+=1
			print ind

		with open( file_event, 'r') as f:
			ind=0
			for line in f:
				raw_data['event'].append(map(float, line.rstrip().lstrip().split(' ')))
				ind+=1
			print ind
		self.raw_data=raw_data

		l=[]
		for list_entry in raw_data['event']:
			l.extend(list_entry)
		set_l=np.array(list(set(l)))
		print np.max(set_l), np.min(set_l)
		return

		with open(file_opinion,'w') as f:
			num_user=len(self.raw_data['event'])
			for user_id,time_point_row,event_row in zip(range(num_user),self.raw_data['time'],self.raw_data['event']):
				for time_point,event in zip(time_point_row,event_row):
					str_to_write=str(user_id+1)+' '+str(time_point)+' '+str(event)+'\n'
					f.write(str_to_write)

		with open(file_node,'w') as f:
			for user_id in range(num_user):
				f.write(str(user_id+1)+'\n')


		#---------------------------------------------book data-----------------------------
		# with open(file_opinion,'w') as f:
		# 	for time_point,event in zip(self.raw_data['time'][0],self.raw_data['event'][0]):
		# 		str_to_write='1 '+str(time_point)+' '+str(event)+'\n'
		# 		f.write(str_to_write)
		#------------------------------------------------------------------------------------
	def get_nodes(self,file_name):
		with open(file_name,'r')  as f:
			nodes=np.array([int(node)-1 for node in f])
		np.sort(nodes)
		self.nodes=nodes
		self.num_node=self.nodes.shape[0]

		# print np.min(nodes)
		# print np.max(nodes)
		# print np.unique(nodes).shape
		# print nodes.shape
		# plt.plot(self.nodes)
		# plt.show()

	def get_opinions(self,file_name):
		def map_opinion(line):
			user,time,opinion = line.split(' ')
			user=int(user)-1
			time=float(time)
			opinion=float(opinion)

			return [user,time,opinion]

		def normalize(arr,low=0,high=1):
			a=np.min(arr)
			b=np.max(arr)
			# return ((arr-a)/(b-a))*10
			return np.array([ (float(t-a)/(b-a) )*(high-low)+low for t in arr ])

		with open(file_name,'r')  as f:
			msg=np.array([ map_opinion(line) for line in f ])
		print msg.shape
		user =  map(int,msg[:,0].flatten())
		# print np.min(user)
		# print np.max(user)
		# print np.unique(user).shape
				
		time = map(float,msg[:,1].flatten())
		# print np.min(time)
		# print np.max(time)
		# plt.plot(time)
		# plt.show()

		time=normalize(time,0,10)
		# print type(time)
		# np.sort(time)
		# plt.plot(time ) 
		# plt.show()


		opinion = map(float,msg[:,2].flatten())
		# plt.plot(opinion)
		# plt.show()
		opinion=normalize(opinion,-1,1)
		# plt.plot(opinion)
		# plt.show()
		
		num_msg=msg.shape[0]
		msg[:,1]=time
		msg[:,2]=opinion
		# check whether it changes time in msg
		# plt.plot(msg[:,2].flatten())
		# plt.show()
		self.msg=msg
		self.msg = self.msg[np.argsort( self.msg[:,1] ), :]
		self.num_msg=self.msg.shape[0]
		# plt.plot(self.msg[:,1])
		# plt.show()
		# plt.plot(opinion)
		# plt.show()

	def truncate(self):
		def normalize(arr,low=0,high=1):
			a=np.min(arr)
			b=np.max(arr)
			# return ((arr-a)/(b-a))*10
			return np.array([ (float(t-a)/(b-a) )*(high-low)+low for t in arr ])

		# np.array([[1,2,3],[4,5,6],[7,1,9],[8,2,9]])#
		msg=np.concatenate((self.train,self.test),axis=0)
		# return 
		# plt.plot(msg[:,0].flatten())
		# plt.show()
		# print np.where(msg[:,1].flatten()<5)[0]
		#------------------------------so------------------------------
		# max_user=750
		# msg=msg[np.where(msg[:,0].flatten()<max_user)[0],:]
		# print msg.shape

		
		# self.msg=msg
		# self.num_msg=self.msg.shape[0]
		# print np.unique(self.msg[:,0].flatten()).shape
		# print np.min(self.msg[:,0])
		# print np.max(self.msg[:,0])
		# # print msg
		# # plt.plot(self.msg[:,1].flatten())
		# # plt.show()
		# del self.train
		# del self.test
		# self.nodes=np.array(range(max_user))
		# self.num_node=max_user


		#----------------------------------------lastfm 
		msg=msg[np.where(msg[:,1].flatten()<.8)[0],:]
		print msg.shape

		# msg=msg[np.where(msg[:,1].flatten()>1)[0],:]
		# print msg.shape
		msg=msg[np.where(msg[:,0].flatten()<100)[0],:]
		print msg.shape
		msg[:,1]=normalize(msg[:,1].flatten(),0,10)
		self.msg=msg
		self.num_msg=self.msg.shape[0]
		# print np.unique(self.msg[:,0].flatten()).shape
		# print np.min(self.msg[:,0])
		# print np.max(self.msg[:,0])
		# # print msg
		# plt.plot(self.msg[:,1].flatten())
		# plt.show()
		del self.train
		del self.test
		self.nodes=np.array(range(100))
		self.num_node=100


def print_reverse(file_input,file_output):

	list_of_lines = []
	with open(file_input,'r') as f:
		while True:
			line = f.readline()
			if line:
				list_of_lines.append(line)
			else:
				break
	list_of_lines.reverse()
	with open(file_output,'w') as f:
		for line in  list_of_lines:
			f.write(line)
		
def main():


	file_prefix_list = ['barca','british_election','GTwitter','jaya_verdict', 'JuvTwitter' , 'MlargeTwitter', 'MsmallTwitter', 'real_vs_ju_703', 'trump_data', 'Twitter' , 'VTwitter']
	for file_name in file_prefix_list:
		file_src = '../Cherrypick_others/Data_opn_dyn_python/'+file_name+'_10ALLXContainedOpinionX.mat.msg'
		file_dest = '../Cherrypick_others/Data_opn_dyn_python/'+file_name+'.txt'
		print_reverse( file_src, file_dest)
	return 





	flag_read_food_data= False # True
	flag_read_reddit_data=False# True
	flag_plot_opinion=False # True
	flag_read_matlab_data = False # True # False
	flag_map_train_data_python = False # True
	generate_dataset_information = False#True #  False #  True
	check_data=False #  True
	merge_train_test=False # True
	preprocess_temporal_data= True
	truncate_data=False#True


	#--------------------------------------------------------------------------------------
	if preprocess_temporal_data:

		# folder_name='lastfm_small'
		folder_name='so'
		# folder_name='mimic2'
		# folder_name='book_data'
		file_path = '../dataset_extra/'+folder_name+'/'
		data=data_preprocess()
		data.read_from_time_events( file_path+'time.txt', file_path+'event.txt', file_path+'opinions.txt',file_path+'nodes.txt')

	if truncate_data:
		file_prefix='lastfm'
		data_file='../Cherrypick_others/Data_opn_dyn_python/'+file_prefix+'_10ALLXContainedOpinionX.obj'
		data=load_data(data_file)	
		data.truncate()
		# return 
		data.split_data(train_fraction=0.9)
		write_file='../Cherrypick_others/Data_opn_dyn_python/'+file_prefix+'_tr_10ALLXContainedOpinionX.obj'
		save(data,write_file)


	if flag_read_food_data:
		data=data_preprocess()

		
		#-------------------------------Food data --------------------------------------------
		# file_prefix='lastfm_small'
		# file_directory='../dataset_extra/'+file_prefix+'/'
		# data.get_nodes(file_directory+'nodes.txt')
		# # return 
		# data.get_opinions(file_directory+'opinions.txt')
		# # return 
		# data.split_data(train_fraction=0.9)

		# print 'Num Node:',data.nodes.shape[0]
		# print 'Num Train:',data.train.shape[0]
		# print 'Num Test:', data.test.shape[0]

		# write_file='../Cherrypick_others/Data_opn_dyn_python/'+file_prefix+'_10ALLXContainedOpinionX.obj'
		# save_data(data, write_file)	
		#--------------------------------------------------------------------------------------


		#-------------------------------Food data --------------------------------------------
		file_directory='../dataset_extra/Twitter_data/Food/'
		# food_data.get_adj_graph_from_dict(file_directory+'edgelist.txt')
		data.get_nodes(file_directory+'nodelist.txt')
		data.get_opinions(file_directory+'opinion.txt')
		data.split_data(train_fraction=0.9)
		write_file='../Cherrypick_others/Data_opn_dyn_python/food_10ALLXContainedOpinionX.obj'
		save_data(data, write_file)	

		print 'Num Node:',data.nodes.shape[0]
		print 'Num Train:',data.train.shape[0]
		print 'Num Test:', data.test.shape[0]


	if merge_train_test:
		file_prefix='reddit'
		file_suffix='_10ALLXContainedOpinionX.obj'
		input_file='../Cherrypick_others/Data_opn_dyn_python/'+file_prefix+file_suffix
		obj=load_data(input_file)
		obj.merge()
		write_file='../Cherrypick_others/Data_opn_dyn_python/'+file_prefix+'_full'
		save_data(obj, write_file)

	if check_data:
		def get_data(index,data):
			return np.concatenate((data.train[:,index], data.test[:,index]), axis=0)
		file_suffix='_10ALLXContainedOpinionX.obj'
		file_prefix='../Cherrypick_others/Data_opn_dyn_python/'
		file_name = file_prefix+ 'so_tr' + file_suffix
		data = load_data(file_name)



		print 'Num Node:',data.nodes.shape[0]
		print 'Num Train:',data.train.shape[0]
		print 'Num Test:', data.test.shape[0]
		return 
		data_plot=get_data(2,data)
		# print data_plot[:10]
		plt.plot(data_plot)
		plt.show()



	if flag_read_reddit_data:
		data=data_preprocess()
		file_directory='../dataset_extra/extra_data/Reddit/'
		file_suffix='_10ALLXContainedOpinionX.obj'
		# data.map_nodes(file_directory+'nodelist.txt', file_directory+'nodelist_mapped.txt')
		# data.map_edges(file_directory+'nodelist.txt', file_directory+'edgelist.txt',file_directory+'edgelist_mapped.txt')
		# data.map_opinions(file_directory+'nodelist.txt', file_directory+'opinion.txt',file_directory+'opinion_mapped.txt')
		

		# ext='_mapped.txt'
		# data.get_nodes(file_directory+'nodelist'+ext)
		# data.get_adj_graph_from_dict(file_directory+'edgelist'+ext)
		# data.get_opinions(file_directory+'opinion'+ext)
		# data.split_data(train_fraction=0.9)
		
		write_file='../Cherrypick_others/Data_opn_dyn_python/reddit'+file_suffix
		# save_data(data, write_file)	
		data=load_data(write_file)
		m=np.concatenate((data.train[:,2], data.test[:,2]), axis=0)
		plt.plot(m)
		plt.show()


	


	if flag_plot_opinion:
		list_of_file_prefix =   ['jaya_verdict', 'JuvTwitter' , 'MlargeTwitter','MsmallTwitter', 'real_vs_ju_703', 'trump_data' ,'Twitter','VTwitter']
		for file_prefix in list_of_file_prefix:
			plot_opinion(file_prefix)
	if generate_dataset_information:
		generate_dataset_info( directory = '../Cherrypick_others/Data_opn_dyn_python/to_send/' , file_ext = '.obj.pkl' )

	if flag_read_matlab_data:
		path = '../Cherrypick_others/Data_opn_dyn_python/'
		filename = path + 'VTwitter_10ALLXContainedOpinionX'
		read_file = filename + '.mat'
		write_file = filename + '.obj'
		fraction_of_train = .9
		dp = data_preprocess(read_file)
		dp.read_data()
		dp.split_data(fraction_of_train)
		save_data(dp , write_file )
	if flag_map_train_data_python:
		directory = '../Cherrypick_others/Data_opn_dyn_python/'
		list_of_files = os.listdir( directory )
		for file in list_of_files :
			# print "------------------"
			# print file
			if file.endswith('.obj.pkl'):
				# print True
				# print directory + file[:-4]
				# return
				obj = load_data( directory + file[:-4] )
				obj.train[:,0] -= 1 
				obj.test[:,0] -= 1
				file_to_write = directory + 'new/'+ file[:-4]
				# print file_to_write 
				save( obj, file_to_write )


		
if __name__ == "__main__":
	main()

