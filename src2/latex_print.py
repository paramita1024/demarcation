from shutil import copyfile
from scipy.interpolate import CubicSpline
import datetime
import os 
import sys
import matplotlib.pyplot as plt
import numpy.random as rnd
import time
from slant import slant 
from data_preprocess import *
from myutil import *

def print_image_in_latex_v4( file_to_write, Image_file_prefix, Image_file_suffix, num_w, num_l,figure_caption):
	with open(file_to_write,'a') as file:
		file.write('\\begin{figure}[!!t]\n')
		file.write('\\centering\n')
		# section = text
		for w_idx in range( num_w):	
			for lamb_idx in range( num_l):
				# parts of image
				for ext in range(20): 
					plot_file= Image_file_prefix + str(ext) + Image_file_suffix
					file.write( '\\subfloat{ \t\\includegraphics[scale=0.25]{' + plot_file + '}}\n')
					if ( ext%3 == 2):
						file.write('\\vspace{-3mm}\n')
			# file.write('\\vspace{-3mm}\n')

			# 	# if lamb_idx%5 == 4:
			# 	plot_file= Image_file_prefix + '_w'+str(w_idx)+'v0l'+str(l_idx) + Image_file_suffix
			# 	file.write( '\\subfloat{ \t\\includegraphics[scale=0.25]{' + plot_file + '}}\n')
			# 	if ( lamb_idx%5 == 2):
			# 		file.write('\\vspace{-3mm}\n')
			# file.write('\\vspace{-3mm}\n')
		file.write('\\caption{'+figure_caption+'}\n')
		file.write('\\end{figure}\n')
def print_image_in_latex_v1( file_prefix, fr_list, cpk_lm_list, lm_list ):
	file_to_write='../opinion_dynamics_others/write_ups/Plot_files.tex'
	with open(file_to_write,'a') as file:
		for f in fr_list:
			for cpk_l in cpk_lm_list:
				for l in lm_list:
					plot_file='../Plots/Plots_with_lambda/Time_vs_MSE/all_combo/'+ file_prefix+'.fraction.'+ str(f) + '.cpk_lambda.'+str(c)+'.lambda.'+ str(l)+'.png' 
					file.write( '\\begin{figure}[h]\n') #\label{online}' )
					file.write( '\\includegraphics[width=\\linewidth,keepaspectratio]{' + plot_file + '} ')
					file.write( '\\end{figure}' )

def print_image_in_latex_v2( file_prefix, fr_list, lm_list ):
	file_to_write='../opinion_dynamics_others/write_ups/Plot_files.tex'
	with open(file_to_write,'a') as file:
		for f in fr_list:
			for l in lm_list:
				plot_file='../Plots/Plots_with_lambda/Time_vs_MSE/tuned_cpk/'+ file_prefix+'.fraction.'+ str(f) + '.lambda.'+ str(l)+'.png' 
				file.write( '\\begin{figure}[h]\n') #\label{online}' )
				file.write( '\\includegraphics[width=\\linewidth,keepaspectratio]{' + plot_file + '}\n ')
				file.write( '\\end{figure}\n\n\n' )

def print_image_in_latex_v3( file_prefix, fr_list ):
	file_to_write='../opinion_dynamics_others/write_ups/Plot_files.tex'
	with open(file_to_write,'a') as file:
		for f in fr_list:
		
			plot_file='../Plots/Plots_with_lambda/Time_vs_MSE/tuned_over_lambda/'+ file_prefix+'.fraction.'+ str(f) + '.png' 
			file.write( '\\begin{figure}[h]\n') #\label{online}' )
			file.write( '\\includegraphics[width=\\linewidth,keepaspectratio]{' + plot_file + '} ')
			file.write( '\\end{figure}' )
def print_image_in_latex_sequential_way(): # contiguous figures
	file_to_write='../opinion_dynamics_others/write_ups/Plot_files.tex'
	folder = '../opinion_dynamics_others/write_ups/Fig'
	if os.path.exists(file_to_write):
		os.remove(file_to_write)
	with open(file_to_write,'a') as file:
			
		for directory in os.listdir( folder ):
			if directory.split('_')[1] == 'cpk':
				# file.write('Here we attach the results while we tune over cherrypick\n')
				file.write('\\section{Tune over cherrypick}\n')
			else:
				file.write('\\section{Tune over nowcasting}\n')
				# file.write('Here we attach the results while we tune each method at nowcasting result\n')
			for files in os.listdir( folder + '/'+directory):
				
				plot_file='Fig/' + directory + '/' + files
				file.write( '\\begin{figure}[bp!]\n') #\label{online}' )
				file.write( '\\includegraphics[width=\\linewidth,keepaspectratio]{' + plot_file + '}\n')
				file.write( '\\caption{'+files.strip('.png')+ '}')
				file.write( '\\end{figure}\n' )


def print_image_in_latex_using_subfig(): # contiguous figures
	file_to_write='../opinion_dynamics_others/write_ups/Plot_files.tex'
	folder = '../opinion_dynamics_others/write_ups/Fig'
	if os.path.exists(file_to_write):
		os.remove(file_to_write)
	with open(file_to_write,'a') as file:
			
		for directory in os.listdir( folder ):
			file.write('\\begin{figure}[ht!]\n')
			file.write('\\centering\n')
			section = 'Sentiment prediction performance using a 10$\\percent$ held-out set for each real-world dataset.'
			section+= '  Performance is measured in terms of mean squared error (MSE) on the sentiment value.'
			section +=  'For each message in the held-out set, we predict the sentiment value m given the history up to T hours before the time of the message,'
			section += 'for different values of T. Nowcasting corresponds to T = 0 and forecasting to T $ > $ 0. '
			section += 'The sentiment value m $\\in$ (-1, 1) and the sentiment polarity sign (m) $\\in$ \\{-1, 1\\}.' 
			if directory.split('_')[1] == 'cpk':
				# file.write('Here we attach the results while we tune over cherrypick\n')
				section += 'We have tune cherrypick results over $\\lambda$ here.'
			else:
				section += 'We have tuned $\\lambda$ for each method based on  nowcasting performance.'
			
				# file.write('Here we attach the results while we tune each method at nowcasting result\n')
			for files in os.listdir( folder + '/'+directory):
				
				plot_file='Fig/' + directory + '/' + files
				file.write( '\\begin{subfigure}{.4\\linewidth}\n') #\label{online}' )
				file.write( '\t\\includegraphics[scale=0.25]{' + plot_file + '}\n')
				file.write( '\t\\caption{'+files.strip('.png')+ '}\n')
				file.write( '\\end{subfigure}\n' )
			file.write('\\caption{'+section+'}\n')
			file.write('\\end{figure}\n')


def print_image_in_latex_using_subfig_v1( file_to_write , text, directory ): # contiguous figures
	
	
	# if os.path.exists(file_to_write):
	# 	os.remove(file_to_write)
	directory += text + '/'
	with open(file_to_write,'a') as file:
		file.write('\\begin{figure}[ht!]\n')
		file.write('\\centering\n')
		section = text
		for files in os.listdir(directory):
			plot_file='Fig/' + text + '/' + files
			file.write( '\\begin{subfigure}{.4\\linewidth}\n') #\label{online}' )
			file.write( '\t\\includegraphics[scale=0.25]{' + plot_file + '}\n')
			file.write( '\t\\caption{'+files.strip('.png')+ '}\n')
			file.write( '\\end{subfigure}\n' )
		file.write('\\caption{'+section+'}\n')
		file.write('\\end{figure}\n')
# def print_image_in_latex( file_prefix ):
# 	fr_list = [0,1]
# 	cpk_lm_list = [0,1]
# 	lm_list = [0,1]
# 	# print_image_in_latex_v1( file_prefix, fr_list, cpk_lm_list, lm_list)
# 	print_image_in_latex_v2( file_prefix, fr_list, lm_list )
# 	# print_image_in_latex_v3( file_prefix, fr_list )

def print_for_kile( data, row_titles, column_titles):
	row_title_with_replacement = []
	for row in row_titles :
		row_title_with_replacement.append( row.replace('_',' '))
		# row = row.replace('_',' ')
	row_titles = row_title_with_replacement
	col_title_with_replacement = []
	for col in column_titles :
		col_title_with_replacement.append( col.replace('_',' '))
		# row = row.replace('_',' ')
	column_titles = col_title_with_replacement



	print '\\begin{center}\n\\begin{tabular}{|c|c|c|c|}\n\\hline'
	header_str = 'Dataset '
	for column in column_titles:
		header_str += ' & '+ column
	# print 'Dataset & MSE & FR \\\\'
	print header_str + '\\\\'
	print '\\hline \n'
	print_str = ''
	# print type( data[0])
	for row,data_row in zip(row_titles, data) :
		print_str += ( row + ' & ' + ' & '.join( map( str , data_row )))
		print_str += '\\\\\n'
		# print ' ' + str( result_dict['name']) + ' & ' + str( result_dict['MSE']) + ' & ' + str( result_dict['FR']) +' \\\\' 
	print print_str + '\\\\'
	print 'hline'
	print '\\end{tabular}'
	print '\\end{center}'

# def find_index( index, num_of_elm ):
# 	param_index = np.zeros( len( num_of_elm ))
# 	for i in range( len( num_of_elm ) -1 ):
# 		param_index[i] = index/ num_of_elm[i]
# 		index = index % num_of_elm[i]
# 	param_index[-1] = index 
# 	return param_index



#************************************************************************



def list_images_for_latex( file_prefix_list, file_to_write, Image_file_prefix,plot_file_suffix1, plot_file_suffix2):
	# Image_file_prefix=Image_file.split('_slant')[0]
	# Image_file_suffix=Image_file.split('_slant')[1]
	file_not_to_include=['trump_data','GTwitter','VTwitter','MlargeTwitter','Twitter']
	file_count=0
	with open(file_to_write,'w') as file:
		file.write('\\begin{figure}[!!t]\n')
		# file.write('\\hspace{-20mm}\n')
		file.write('\\centering\n')
		# section = text
		buffer=0
		for file_prefix in file_prefix_list:	
			if file_prefix not in file_not_to_include:
				plot_file= Image_file_prefix+file_prefix
				file.write( '\\subfloat{ \t\\includegraphics[scale=0.20]{' + plot_file + plot_file_suffix1 + '}}\n')
				file.write( '\\subfloat{ \t\\includegraphics[scale=0.20]{' + plot_file + plot_file_suffix2 + '}}\n')
				buffer+=1
				if ( buffer == 2):
					file.write('\\vspace{-3mm}\n')
					buffer=0
				file_count+=1
				# if file_count==6:
				# 	file.write('\\caption{Here we plot MSE Vs time span for 6 datasets}\n')
				# 	file.write('\\end{figure}\n')
				# 	file_count=0
				# 	file.write('\\begin{figure}[!!t]\n')
				# 	file.write('\\centering\n')
		# figure_caption=file_prefix.replace('_', ' ')
		file.write('\\caption{Here we plot MSE Vs time span for rest datasets}\n')
		file.write('\\end{figure}\n')
		# file.write('\\paragraph')

def write_table_sanitize_test( file_to_read, file_prefix_list):
	def get_substr(v):
		return str(round(v,3))#[:6]
	result=load_data(file_to_read)

	print '\\begin{table}[h!]'
	print '\t\\begin{center}'
	print '\t\t\\caption{Performance after sanitizing test sets}'
	print '\t\t\\label{tab:Table1}'
	print '\t\t\\begin{tabular}{|l|l|l|l|l|l|l|l|l|}'
	print '\t\t\\hline'
	print '\t\t\\& \\multicolumn{4}{|l|}{MSE} & \\multicolumn{4}{|l|}{FR}\\'
	print '\t\t\tDataset & RCPK & RCPK-ST & CPK & CPK-ST & RCPK & RCPK-ST & CPK & CPK-ST \\\\'
	print '\t\t\t\\hline'

	for file in file_prefix_list:
		# if file=='MlargeTwitter':
		# 	print result['Robust_cherrypick'][file]
		# 	print result['cherrypick'][file]
		data=[]
		for measure in ['MSE', 'FR']:
			for method in ['Robust_cherrypick' ,'cherrypick']:
				for ext in ['', '_san']:
					# print method, measure+ext
					data.append(result[method][file][measure+ext])
		# data=np.array([result['Robust_cherrypick'][file]['FR_san'], result['Robust_cherrypick'][file]['FR'], result['cherrypick'][file]['FR_san'], result['cherrypick'][file]['FR']])
		# data=[result['Robust_cherrypick'][file]['MSE'],result['Robust_cherrypick'][file]['MSE_san'], result['cherrypick'][file]['MSE'], result['cherrypick'][file]['MSE_san']]
		# data=[result['Robust_cherrypick'][file]['FR_san'], result['Robust_cherrypick'][file]['MSE'], result['cherrypick'][file]['MSE_san'], result['cherrypick'][file]['MSE']]
		print_str = ( file.replace('_', ' ') + ' & ' + ' & '.join( map( get_substr , data )))
		print_str += '\\\\'
		print '\t\t\t'+print_str+'\n\t\t\t'+'\\hline'+'\n'
		
	# print print_str
	print '\t\t\t\\hline'
	print '\t\t\\end{tabular}'
	print '\t\\end{center}'
	print '\\end{table}'

	# print '\\paragraph{Description}'
	# print 'For both Robust Cherrypick and Cherrypick method, original test set is filtered again \
# to find only endogenious test messages. MSE is recomputed on that sanitized test set. \
# over the full data set including both training and test set to select only .8 fraction most \
# endogenious message of the full data set. The subset of original test set that intersects \
# with this endogenious subset is considered as sanitized test set. Results sre noted as RCPK-ST and CPK-ST where \
# original results are noted as RCPK and CPK.'


def main():

	# obj=load_data('barca.pkl')
	# # idx=np.arange(10000)[0:1000]
	# plt.plot(obj['t'],obj['I'])
	# plt.plot(obj['t'],obj['e'])
	# plt.show()
	# return
	list_of_windows = np.array([.4]) # np.linspace(0.05,1,20)
	time_span_input_list = np.linspace(0,.5,6) 
	file_prefix_list =   ['barca','british_election','GTwitter','jaya_verdict', 'JuvTwitter' , 'MlargeTwitter','MsmallTwitter','real_vs_ju_703', 'trump_data','Twitter','VTwitter']
	 # 'real_vs_ju_703','trump_data' ,
	
	print_slant_result= False # True # False # True
	print_image_in_latex_flag = False 
	print_intensity_latex =False #True
	print_opinion_lambda=False#True#True
	print_plots_latex=False # True # False # True
	write_table_sanitize_test_flag=True # True
	print_combined_variation_fraction=False # True

	if print_combined_variation_fraction:
		for file in file_prefix_list:
			if file not in ['GTwitter','MlargeTwitter','trump_data','Twitter','VTwitter']:
				# print '\\subfloat{ 	\\includegraphics[scale=0.15]{FIG_new/MSE.jpg\}\}'
				print('\\subfloat{ \t\\includegraphics[scale=0.15]{FIG_new/' + file + '_combined_MSE.jpg' + '}}')
		print('\\vspace{-3mm}')
		for file in file_prefix_list:
			if file not in ['GTwitter','MlargeTwitter','trump_data','Twitter','VTwitter']:
				# print '\\subfloat{ 	\\includegraphics[scale=0.15]{FIG_new/MSE.jpg\}\}'
				print('\\subfloat{ \t\\includegraphics[scale=0.15]{FIG_new/' + file + '_combined_FR.jpg' + '}}')
		

	if write_table_sanitize_test_flag:
		file_to_read='../result_sanitize_test/f0.8t0.2_MSE_FR'
		# file_prefix_list.remove('GTwitter')
		file_prefix_list.remove('MlargeTwitter')
		file_prefix_list.remove('trump_data')
		# file_prefix_list.remove('VTwitter')
		write_table_sanitize_test( file_to_read, file_prefix_list)


	if print_plots_latex:
		# file_to_write='../paper/0511expModelingMSEnew.tex'
		# file_to_write='../../../Dropbox/Others/Paramita/paper/0511expModelingNew.tex'
		# file_to_write='../paper_working_copy/0511expVarFracRcpkMSE.tex'
		file_to_write='../../../Dropbox/Others/Paramita/paper/0511expVarFracNew.tex'
		if os.path.exists(file_to_write):
			os.remove(file_to_write)
		Image_file_pre='FIG_new/'
		# Image_file_post='_Robust_cherrypick_MSE.jpg'
		# Image_file_post1='_slant_tuned_0.jpg'
		# Image_file_post2='_final.jpg'
		Image_file_post1='_cherrypick_MSE.jpg'
		Image_file_post2='_Robust_cherrypick_MSE.jpg'
		list_images_for_latex( file_prefix_list,file_to_write,Image_file_pre,Image_file_post1,Image_file_post2 )

	
	if print_slant_result :
		for file_prefix in file_prefix_list : 
			print_slant_results( file_prefix  )

	if print_image_in_latex_flag:
		# print_image_in_latex_sequential_way()
		print_image_in_latex_using_subfig()
		# file_index_set = [ 0,1,4,7]
		# for index in file_index_set:
		# 	print_image_in_latex( file_prefix_list[index] )
	
	if print_intensity_latex:
		# file_to_write='../opinion_dynamics_others/write_ups/Opinions_Intensity.tex'
		# directory  = '../opinion_dynamics_others/write_ups/Fig/' 
		
		
		# print_image_in_latex_using_subfig_v1( file_to_write , 'Intensity', directory )
		
		# print_image_in_latex_using_subfig_v1( file_to_write , 'Opinions', directory )
		# file_prefix_list = [ file_prefix_list[0]]
		file_to_write='../opinion_dynamics_others/write_ups/Intensity.tex'
		if os.path.exists(file_to_write):
			os.remove(file_to_write)
		for file_prefix in file_prefix_list:
			Image_file_prefix='Fig/Intensities/'+file_prefix+'_window_'
			Image_file_suffix='_Intensities.eps'
			num_windows=20
			figure_caption=file_prefix.replace('_', ' ')
			print_image_in_latex_v4( file_to_write, Image_file_prefix, Image_file_suffix, num_windows, figure_caption)

	if print_opinion_lambda:
		file_prefix_list = [ file_prefix_list[0]]
		# file_to_write='../opinion_dynamics_others/write_ups/Opinions_lambda.tex'
		# if os.path.exists(file_to_write):
		# 	os.remove(file_to_write)
		num_w=1#4
		num_l=1 # 5
		for file_prefix in file_prefix_list:
			file_to_write='../opinion_dynamics_others/write_ups/Opinions_w0v0l0_true_opinion_exact_pred_'+file_prefix+'.tex'
			if os.path.exists(file_to_write):
				os.remove(file_to_write)
			Image_file_prefix='Fig/w0v0l0_barca/barca_w0v0l0_window_0_Exact_Opinions_ext_'
			Image_file_suffix='.eps'
			
			figure_caption=file_prefix.replace('_', ' ')
			print_image_in_latex_v4( file_to_write, Image_file_prefix, Image_file_suffix, num_w, num_l, figure_caption)
	
	

if __name__=='__main__':
	main()

