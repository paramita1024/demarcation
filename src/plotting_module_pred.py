from myutil import *
from data_preprocess import data_preprocess
import pickle
import numpy as np
from eval_slant_module import *
import matplotlib.pyplot as plt

class results:
    def __init__(self,file_prefix,file_name):
        self.file_prefix=file_prefix
        self.file_name=file_name
        
    def save_pred_matlab(self,x_axis):
        if x_axis == 'time':
            res_file=self.file_name['res_vs_time']
        if x_axis == 'frac':
            res_file=self.file_name['res_vs_frac']
        res=load_data(res_file)
        # print type(res['y-axis']['MSE'])
        # print res['y-axis']['MSE'].shape
        res['x-axis']=np.array(res['x-axis'])
        len_x=res['x-axis'].shape[0]
        # print res['x-axis']
        arr_MSE=np.concatenate((res['x-axis'].reshape(1,len_x),res['y-axis']['MSE']),axis=0)
        write_txt_file(arr_MSE.transpose(),res_file[:-8]+'MSE.txt')
        arr_FR=np.concatenate((res['x-axis'].reshape(1,len_x),res['y-axis']['FR']),axis=0)
        write_txt_file(arr_FR.transpose(),res_file[:-8]+'FR.txt')
        
def main():

    file_prefix_list=['barca','british_election','GTwitter','jaya_verdict', 'JuvTwitter' , 'MlargeTwitter','MsmallTwitter', 'real_vs_ju_703', 'trump_data' ,'Twitter','VTwitter']
    #-------------INPUT PARAMETERS----------------
    for file_prefix in file_prefix_list:
        if file_prefix not in ['MlargeTwitter','trump_data']:
            # file_prefix=file_prefix_list[int(sys.argv[1])]
            file_name={}
            file_name['res_vs_time']='../result_performance_forecasting_pkl/'+file_prefix+'.selected'
            file_name['res_vs_frac']='../result_variation_fraction_pkl/'+file_prefix+'.selected'
            result_saver_obj=results(file_prefix,file_name)
            result_saver_obj.save_pred_matlab('time')
            result_saver_obj.save_pred_matlab('frac')


if __name__=="__main__":
	main()