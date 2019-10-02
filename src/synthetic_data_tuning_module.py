import copy
import sys
import time
from Robust_Cherrypick import Robust_Cherrypick 
from cherrypick import cherrypick
from math import sqrt
from myutil import *
import matplotlib.pyplot as plt
from slant import slant
from numpy import linalg as LA 
import numpy as np
import numpy.random as rnd
from math import ceil
import networkx as nx
import pickle
from myutil import *
from synthetic_data import synthetic_data


class synthetic_data_tuning_module:
	def __init__(self,synthetic_obj):

	
	def eval_slant(self,lamb):
		data_dict={'train':synthetic_obj.train,'test':synthetic_obj.test,'edges':synthetic_obj.edges}
		slant_obj=slant( obj=data_dict,init_by='dict',data_type='real',tuning_param=[self.w,self.v,lamb],int_generator=1)
		pred_param=slant_obj.estimate_param()
		pred_res=slant_obj.predict( num_simulation=1,time_span_input=0)
	return pred_param,pred_res

	


def main():


if __name__=="__main__":
	main()

