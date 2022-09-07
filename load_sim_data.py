import pickle
import xlsxwriter
import openpyxl

entry = 'emer'

with open(r"./Results/"+entry+"/config_exp_7_Active.pkl", "rb") as input_file:
	sim_data = pickle.load(input_file)



print(sim_data)

print('Number of interations: ',  len(sim_data['user_log']))

# Reproduce sim video/frames

# plot coverage over time

# highlight when interaction was triggered.