import pickle
import xlsxwriter
import matplotlib.pyplot as plt

'''
	Plot the coverage achieved in a trial with markers for when 
	a command was triggered (only active trials).

'''

# entry = 'emer'
entry = '20220907T171125'

with open(r"./Results/"+entry+"/config_exp_7_Active.pkl", "rb") as input_file:
	sim_data = pickle.load(input_file)



print(sim_data)

print('Number of interations: ',  len(sim_data['user_log']))

# plot coverage over time

fig, ax = plt.subplots( figsize=(10,5), dpi=120, facecolor='w', edgecolor='k')

ax.plot(sim_data['coverage'])
ax.set_ylim([0,1.0])
ax.grid()

ax.set_ylabel('Coverage Percentage')
ax.set_xlabel('Timesteps')

colour_dict = {'left': 'green', 'right': 'red', 'up': 'blue', 'down': 'orange'}

if 'user_log' in sim_data:

	# plot user input
	for entry in sim_data['user_log']:

		plt.axvline(x=entry[1], color=colour_dict[entry[0]], linestyle='dashed', lw=1, label=entry[0])


# place the legend outside
plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
plt.show()


