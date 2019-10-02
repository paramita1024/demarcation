from myutil import *
import os
# from slant import results
class results : 
	def __init__(self, result_val_MSE, result_val_FR, predicted_val, original_val):
		self.result_val_MSE = result_val_MSE 
		self.result_val_FR = result_val_FR 
		self.predicted_val = predicted_val
		self.original_val = original_val
# write another code to read results from dictionary
def print_for_kile( res_list ):

	print '\\begin{center}\n\\begin{tabular}{|c|c|c|}\n\\hline'
	print 'Dataset & MSE & FR \\\\'

	print '\\hline \n'
	
	for result_dict in res_list :
		print ' ' + str( result_dict['name']) + ' & ' + str( result_dict['MSE']) + ' & ' + str( result_dict['FR']) +' \\\\' 
	
	print '\\hline'

	print '\\end{tabular}'

	print '\\end{center}'

def get_result_info_from_file( path_to_file, flag_send_all = False  ):
	if flag_send_all:
		pass
	else:
		res_obj = load_data( path_to_file[:-4])
		# print " the result obj  consists of time span list followed by a list of " + str( len(result_obj)-1)+ " result objects"
		time_span_input_list = res_obj[0]
		desired_index = 3
		dataset_title = path_to_file.split('/')[-1].split('_10ALLXContainedOpinionX')[0].replace('_',' ')
		# res_obj[desired_index]['name'] = dataset_title
		# return res_obj[ desired_index ]
		obj = res_obj[desired_index]
		res_dict = {}
		res_dict['name'] = dataset_title
		res_dict['MSE'] = obj.result_val_MSE
		res_dict['FR'] = obj.result_val_FR
		return res_dict


		

		# print " time span inputs are " 
		# print time_span_input_list

		# res_index = 1 
		# return

		# for res_obj in result_obj[1:]:
		# 	# print " ------------------------ Result object  " + str( res_index ) 

		# 	# print "MSE  " + str(res_obj.result_val_MSE)
			
		# 	# print "FR  " + str(res_obj.result_val_FR)
			
		# 	plt.plot(np.mean( res_obj.predicted_val , axis = 1 ), "r")
		# 	plt.plot(res_obj.original_val, "b")
		# 	plt.xlim((200,300))
		# 	plt.show() 
			
		# 	plt.savefig(file_to_save_prefix + "_opinion for 100 users "+ str(res_index)+".png")
			 

def main():

	flag_plot_multiple_files = True
	if flag_plot_multiple_files : 
		directory = '../result/' # create 
		list_of_files = os.listdir( directory )
		res_list = []
		for file in list_of_files :
			path_to_file = directory + file 
			# print path_to_file
			result_info = get_result_info_from_file( path_to_file, flag_send_all  = False) # define 
			res_list.append( result_info )
		print_for_kile( res_list )
		return 


	file_prefix = 'MsmallTwitter'
	path = '../Cherrypick_others/result/' # create 
	file = file_prefix + '_10ALLXContainedOpinionX'
	file_to_load = path + file + '.res'

	file_to_save_prefix = path + file 

	result_obj = load_data(file_to_load)
	# print "MSE  " + str(result_obj.result_val_MSE)
	# print "FR  " + str(result_obj.result_val_FR)
	# index  = np.arange(200)
	# plt.plot(result_obj.predicted_val[index], "r")
	# plt.plot(result_obj.original_val[index], "b")
	# plt.show()
	print " the result obj  consists of time span list followed by a list of " + str( len(result_obj)-1)+ " result objects"
	time_span_input_list = result_obj[0]
	print " time span inputs are " 
	print time_span_input_list

	res_index = 1 
	# return
	for res_obj in result_obj[1:]:
		print " ------------------------ Result object  " + str( res_index ) 

		print "MSE  " + str(res_obj.result_val_MSE)
		
		print "FR  " + str(res_obj.result_val_FR)
		
		plt.plot(np.mean( res_obj.predicted_val , axis = 1 ), "r")
		plt.plot(res_obj.original_val, "b")
		plt.xlim((200,300))
		plt.show() 
		
		plt.savefig(file_to_save_prefix + "_opinion for 100 users "+ str(res_index)+".png")
		




if __name__=="__main__":
	main()
