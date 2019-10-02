import os
import time
from myutil import *
from spg_new import spg
from data_preprocess import data_preprocess
import matplotlib.pyplot as plt
from create_synthetic_data import create_synthetic_data
import pickle
from PriorityQueue import PriorityQueue
import math
import numpy.random as rnd
import sys
import numpy as np
from numpy import linalg as LA


class slant_wtout_network:
	def __init__(self,obj,tuning_param):
		self.nodes=obj.nodes
		self.num_node=self.nodes.shape[0]
		self.train = obj.train 
		self.test = obj.test
		self.num_train= self.train.shape[0]
		self.num_test= self.test.shape[0]
		self.var = .1 # 
		self.int_generator='Poisson'
		self.msg_window = tuning_param[0]
		self.lambda_least_square = tuning_param[1]	

	def solve_least_square(self,A,b):
		A_T_b = A.T.dot(b) 
		eye_mat=np.eye( A.shape[1] )
		
		regularizer=(self.lambda_least_square*eye_mat) 
		mat = np.matmul(A.T,A)  + regularizer
		x=LA.solve( mat , A_T_b)
		self.check_target=b
		self.check_prediction=A.dot(x)
		return x[:self.num_node],x[self.num_node:]

	def find_alpha_A(self):
		regression_matrix = np.concatenate((self.msg_user_index,self.opinion_matrix),axis=1)
		all_msg = self.train[:,2]
		self.alpha, self.a = self.solve_least_square( regression_matrix, all_msg )

	def get_sigma_covar_only(self):
		self.find_alpha_A()
		return self.estimate_variance()

	def estimate_intensity_poisson(self):
		self.mu=np.zeros(self.num_node)
		self.b=0
		time_interval=self.train[-1,1]-self.train[0,1]
		for user in self.nodes : 
			self.mu[user] = float(np.where(self.train[:,0]==user)[0].shape[0])/time_interval

	def create_opinion_matrix(self):
		self.msg=np.concatenate((self.train[:,2],self.test[:,2]),axis=0) 
		self.opinion_matrix=np.zeros((self.num_train,self.msg_window))
		for ind in range(1,self.num_train):
			end_ind=max(ind-self.msg_window,0)
			self.opinion_matrix[ind, :min([ind,self.msg_window])]=np.flipud(self.msg[end_ind:ind] )
	def create_initial_data_matrix( self ):
		self.create_opinion_matrix()
		self.msg_user_index=np.zeros((self.num_train,self.num_node))
		for ind in range(self.num_train): 
			self.msg_user_index[ind,int(self.train[ind,0])] = 1


	def estimate_param(self,lamb=None, max_iter=None):
		self.create_initial_data_matrix() 
		if lamb!=None:
			self.lambda_least_square=lamb 
		self.estimate_intensity_poisson() 
		self.find_alpha_A() 
		self.var = self.estimate_variance() 

	def estimate_variance( self ):
		predict_train = self.map_alpha(self.train[:,0])+self.opinion_matrix.dot(self.a)
		return np.mean(( predict_train - self.train[:,2])** 2 )


	def map_alpha(self,list_of_users):
		list_of_alphas=np.zeros(list_of_users.shape[0])
		for user,ind in zip(list_of_users,range(list_of_users.shape[0])):
			list_of_alphas[ind]=self.alpha[int(user)]
		return list_of_alphas

	def get_influence(self,ind):
		if ind <= self.msg_window:
			print 'break. Error\nError\nError\n'
		start=ind-self.msg_window
		return self.a.dot(np.flipud(self.msg[start:ind ])) 

	def set_train( self, train_data ):
		self.train = train_data 
		self.num_train = self.train.shape[0]
		self.create_initial_data_matrix() 

	def get_acc(self,prediction):#mimic
		def map_class(v):
			return ((v+1)/2)*74+1
		pred_cls = map(map_class, prediction)
		true_cls = map(map_class, self.test[:,2])
		corr_pred=0
		for p,t in zip(pred_cls,true_cls):
			if np.absolute(t-p)< 1 :
				corr_pred+=1
		return float(corr_pred)/prediction.shape[0]
	def get_acc_binned(self,prediction):#stack
		# def map_class(v):
		# 	return ((v+1)/2)*21+1
		# pred_cls = map(map_class, prediction)
		# true_cls = map(map_class, self.test[:,2])
		corr_pred=0
		for p,t in zip(prediction,self.test[:,2]):
			if np.absolute(t-p)< .6 :
				corr_pred+=1
		return float(corr_pred)/prediction.shape[0]
	def predict(self, num_simulation, time_span_input, num_msg_poisson_input=None):
		msg_set={}		
		if time_span_input==0: 
			predict_test = self.map_alpha(self.test[:,0])+map(self.get_influence,np.arange(self.num_test)+self.num_train) 
			MSE_loss = get_MSE(predict_test, self.test[:,2])
			FR_loss = get_FR(predict_test, self.test[:,2])
			cls_err= 1-self.get_acc_binned(predict_test)
		else:
			num_msg_poisson=int(time_span_input)
			predict_test = np.zeros( (self.num_test, num_simulation) )
			for msg_index,msg in zip(range(self.num_test),self.test): 
				for sim_no in range(num_simulation):
					msg_set[msg_index],buffer_msg_window=self.simulate_events_poisson(num_msg_poisson_input,msg[1],self.num_train+msg_index)
					predict_test[msg_index,sim_no]=self.alpha[int(msg[0])]+self.a.dot(buffer_msg_window)
					
			mean_predict_test = np.mean( predict_test, axis = 1 ) 
			MSE_loss = get_MSE(mean_predict_test, self.test[:,2])
			FR_loss = get_FR(mean_predict_test, self.test[:,2])

		results = {}
		results['MSE'] = MSE_loss 
		print 'lamb',self.lambda_least_square
		print 'MSE++++++++++++++++++++++',MSE_loss
		print 'cls err', cls_err
		results['FR'] = cls_err
		results['predicted']=predict_test
		results['true_target'] = self.test[:,2]
		results['msg_set']= msg_set
		results['cls_err']=cls_err
		results['check_target']=self.check_target
		results['check_prediction']=self.check_prediction
		return results

	def simulate_events_poisson(self,num_msg,time_curr,msg_index): 
			
		time_span= num_msg/float(np.sum(self.mu))
		start_time=time_curr-time_span

		num_msg_poisson=self.find_num_msg_poisson(time_span)
		msg=np.zeros((num_msg_poisson,3))
		counter=0
		for user in  self.nodes:
			n_msg=int(self.mu[user]*time_span)
			if n_msg!=0:
				msg[counter:counter+n_msg,0]=user
				msg[counter:counter+n_msg,1]=rnd.uniform(low=start_time,high=time_curr,size=n_msg) 
				counter+=n_msg
		msg=msg[np.argsort(msg[:,1]),:]

		buffer_list=list(np.flipud(self.msg[msg_index-self.msg_window:msg_index]))

		for ind,m in zip(range(num_msg_poisson),msg):	
			u=int(m[0])
			x_new= self.alpha[u]+self.a.dot(np.array(buffer_list))
			msg[ind,2] = rnd.normal( x_new , math.sqrt(self.var) )
			buffer_list.pop()
			buffer_list.insert(0,msg[ind,2])

		return msg,buffer_list

	
	def set_parameter(self, obj,flag_dict=False):
		if flag_dict:
			self.mu = obj['mu']
			self.alpha = obj['alpha'] 
			self.A = obj['A']
			self.B = obj['B']
		else:
			self.mu = obj.mu
			self.alpha = obj.alpha 
			self.A = obj.A
			self.B = obj.B

	def set_msg( self, flag_only_train = True,  msg_set = [] ):
		if flag_only_train :
			self.train = msg_set
		self.test = np.array([])
	

	def get_mean_intensity_of_msg( self, time, interval, timestamps ):
		tm_start=time - interval/2.0
		tm_end=time + interval/2.0
		return np.count_nonzero(np.logical_and( timestamps >= tm_start, timestamps <= tm_end))/(float(interval)*self.num_node)
		

	def find_num_msg_poisson(self,time_span_input):
		counter=0
		for user in  self.nodes:
			counter+=int(self.mu[user]*time_span_input)
		return counter
