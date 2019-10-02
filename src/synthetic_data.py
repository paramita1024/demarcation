import time
from Robust_Cherrypick import Robust_Cherrypick 
from cherrypick import cherrypick
from math import sqrt
from myutil import *
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from slant import slant
from numpy import linalg as LA 
import numpy as np
import numpy.random as rnd
from math import ceil
import networkx as nx
import pickle
class synthetic_data:
	def __init__(self):
		pass
		
	def generate_graph( self,generate_str, num_node, num_nbr , num_node_start, num_rep):
		# print 'inside'
		# print generate_str
		if generate_str == 'barabasi_albert':
		
			self.num_node = num_node
			self.nodes = np.arange(self.num_node)
			self.generate_str=generate_str
			self.num_nbr=num_nbr
			G_networkx = nx.barabasi_albert_graph(self.num_node, num_nbr) # check how we can obtain the edges

			self.edges =np.array(nx.adjacency_matrix(G_networkx).todense())
			avg_degree=np.average(np.array([np.count_nonzero(self.edges[node]) for node in self.nodes]))
			print 'avg_degree',avg_degree
			return G_networkx

		if generate_str=='kronecker':
			outfile='outfile'
			self.generate_kronecker(num_node_start, num_rep) # set edges,num_node and nodes 
			self.write_graph(self.edges,outfile)
			G_nx=nx.read_adjlist(outfile)
			nx.draw(G_nx)
			plt.show()
			return G_nx
	
	def basic_graph(self,num_node):
		if num_node==4:
			adj = np.array([[1,0,1,0],[0,1,1,1],[1,1,1,1],[0,0,1,1]])
		if num_node==3:
			adj=np.array([[1,1,0],[1,0,1],[0,1,1]])
			# adj=np.array([[1,1,0],[1,1,1],[0,1,1]])
			# adj=np.array([[1,0,1],[0,1,0],[1,0,1]])
			# adj=np.array([[1,0,0],[0,1,1],[0,1,1]])
			# adj=np.array([[1,1,1],[1,0,0],[1,0,1]])
		return adj

	def generate_kronecker(self,num_node_start, num_rep):
		self.basic_graph_adj = self.basic_graph(num_node_start)
		self.adj=np.array(self.basic_graph_adj)
		for i in range(num_rep):
			self.adj = np.kron(np.array(self.adj),self.basic_graph_adj)
		np.fill_diagonal(self.adj,0)
		self.edges=self.adj 
		self.num_node = self.edges.shape[0]
		print 'num_node', self.num_node
		self.nodes = np.arange(self.num_node)

	def write_graph( self,adj,file):
		with open(file,'w') as f:
			for node,line in zip(self.nodes,adj):
				f.write(str(node)+' '+' '.join(map(str,np.where(line>0)[0]))+'\n')

			
	def generate_parameters(self,max_intensity=0):
		
		self.alpha = rnd.normal(0,1,self.num_node)
		self.mu = rnd.uniform(0,1, self.num_node)
		self.A = np.zeros( (self.num_node, self.num_node) )
		self.B = np.zeros( (self.num_node, self.num_node) )
		for node in self.nodes:
			self.A[node, np.array(self.edges[node],dtype=bool)] = rnd.normal(0,1, np.count_nonzero(self.edges[node]))
			self.B[node, np.array(self.edges[node],dtype=bool)] = rnd.uniform(0,1,np.count_nonzero(self.edges[node]))
			self.B[node,node] = rnd.uniform(0,1,1)
		###################### Alternate way  ######################################################

		# max_intensity= .1#.1 # .05 #
		# intensity_coeff=.32#1.2#.1 # 
		# opinion_coeff=.2
		# opinion_coef_start=0.01
		# self.alpha = opinion_coef_start*rnd.uniform(-1,1, self.num_node)
		# # option A 
		# # self.mu = rnd.uniform(0,max_intensity,self.num_node) 
		# # option B 
		# self.mu = np.zeros(self.num_node)
		# subset_node=rnd.choice(self.num_node, int( self.num_node))
		# self.mu[subset_node]=rnd.uniform(0,max_intensity,subset_node.shape[0])
		# self.A = np.zeros( (self.num_node, self.num_node) )
		# self.B = np.zeros( (self.num_node, self.num_node) )
		
		# for node in self.nodes:
		# 	self.A[node, np.array(self.edges[node],dtype=bool)] = opinion_coeff*rnd.uniform(-1,1, np.count_nonzero(self.edges[node]))
		# 	self.B[node, np.array(self.edges[node],dtype=bool)] = intensity_coeff*rnd.uniform(0,1,np.count_nonzero(self.edges[node]))
		# 	self.B[node,node] = intensity_coeff*rnd.uniform(0,1,1)
		# self.label_int='mu~U(0,'+str(max_intensity)+'),B~U(0,'+str(intensity_coeff)+')'
		# self.label_opn='alpha~'+str(opinion_coef_start)+'*U(-1,1),A~'+str(opinion_coeff)+'*U(-1,1)'

	def init_from_dict_generate_msg(self,data_dict):
		# dict keys() => edges , A B alpha mu w v 
		self.edges=data_dict['edges']
		self.A=data_dict['A']
		self.B=data_dict['B']
		self.alpha=data_dict['alpha']
		self.mu=data_dict['mu']
		self.generate_msg_single(time_span=10,p_exo=0,var=0,flag_check_num_msg =True,max_msg=max_msg,w=dict['w'],v=dict['v'])
	


	def generate_msg(self, plot_file, time_span, option=None,p_exo=None,var=0, noise_param=None, flag_check_num_msg = False,max_msg=0):
		def noise():
			# return rnd.normal( rnd.uniform(-1,1,1), sqrt(noise_param))
			return rnd.normal(0, sqrt(noise_param))

		if noise_param==0:
			noise_param=var
		
		self.time_span=time_span
		dict_obj={'edges':self.edges}#, 'param':param}
		# slant_obj=slant(init_by='dict', data_type='synthetic', obj = dict_obj, tuning_param=[10,10,float('inf')])
		slant_obj=slant(init_by='dict', obj = dict_obj, tuning_param=[1000,10,float('inf')])
		# self.temp=9
		if option=='both_endo_exo':
			# print 'exo'
			self.exo=True
			self.msg, self.msg_end =slant_obj.simulate_events(time_span , self.mu, self.alpha, self.A, self.B,var=var,p_exo=p_exo,noise=noise,flag_check_num_msg = flag_check_num_msg,max_msg=max_msg)
		else:
			# self.msg=slant_obj.simulate_events(self.temp) # time_span , self.mu, self.alpha, self.A, self.B)
			self.msg,temp1,temp2=slant_obj.simulate_events(time_span , self.mu, self.alpha, self.A, self.B,var=var,flag_check_num_msg =flag_check_num_msg,max_msg=max_msg)
		self.num_msg = self.msg.shape[0]
		print 'max msg', np.max(self.msg[:,2])
		# return 
		# self.msg=normalize(self.msg)
		plt.subplot(2,1,1)
		plt.plot(self.msg[:,1], self.msg[:,2]) # ,label=self.label_opn)
		plt.xlabel('time')
		plt.ylabel('opinion')
		plt.legend()
		# plt.ylim(-1,1)
		# plt.show()
		plt.subplot(2,1,2)
		plt.plot(range(self.num_msg), self.msg[:,1])# ,label=self.label_int)
		plt.xlabel('index')
		plt.ylabel('time')
		plt.legend()
		# plt.show()

		plt.savefig(plot_file)
		plt.show()
		plt.clf()
		
		# print self.num_msg

	def generate_msg_single(self, plot_file, time_span,p_exo=None,var=0,flag_check_num_msg = False,max_msg=0,w=None,v=None):
		if w==None or v==None:
			print 'please set w and v '
			return
		self.time_span=time_span
		dict_obj={'edges':self.edges}
		slant_obj=slant(init_by='dict', obj = dict_obj, tuning_param=[w,v,float('inf')])
		self.msg, self.msg_end =slant_obj.simulate_events(time_span , self.mu, self.alpha, self.A, self.B,var=var,p_exo=p_exo,flag_check_num_msg = flag_check_num_msg,max_msg=max_msg)
		self.num_msg = self.msg.shape[0]
		print 'max msg', np.max(self.msg[:,2])
		plt.subplot(2,1,1)
		plt.plot(self.msg[:,1], self.msg[:,2]) # ,label=self.label_opn)
		plt.xlabel('time')
		plt.ylabel('opinion')
		plt.legend()
		plt.subplot(2,1,2)
		plt.plot(range(self.num_msg), self.msg[:,1])# ,label=self.label_int)
		plt.xlabel('index')
		plt.ylabel('time')
		plt.legend()
		plt.savefig(plot_file)
		plt.show()
		plt.clf()
		
	def split_data(self, train_fraction):
		num_tr = int( self.num_msg * train_fraction )
		self.train = self.msg[:num_tr]
		self.test = self.msg[num_tr:]
		if hasattr(self,'msg_end'):
			self.msg_end_tr=self.msg_end[:num_tr]
			self.msg_end_test=self.msg_end[num_tr:]
		# del self.msg

	def clear_msg(self):
		self.msg=None
		self.train=None
		self.test=None

	def read_from_csv_file(self,file_in, datatype):

		with open(file_in, 'r') as f:
			if datatype=='int':
				# nline=0
				# for line in f:

				# 	nline+=1
				# print 'nline',nline
				temp = np.array([ [ int(num) for num in line.split(',') ] for line in f ])
			if datatype=='float':
				temp = np.array([ [ float(num) for num in line.split(',') ] for line in f ])
		if temp.shape[0]==1:
			temp=temp[0]
		return temp

	def init_from_file(self,file_pre):

		self.edges=self.read_from_csv_file(file_pre+'Adj','int')
		self.alpha=self.read_from_csv_file(file_pre+'alpha','float')
		self.mu=self.read_from_csv_file(file_pre+'mu','float')
		self.A=self.read_from_csv_file(file_pre+'A','float')
		self.B =self.read_from_csv_file(file_pre+'B','float')
		self.num_node = self.edges.shape[0]
		print 'num_node', self.num_node

		self.nodes = np.arange(self.num_node)
