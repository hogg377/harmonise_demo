import pickle

entry = '20220725T190804'

with open(r"Results/"+entry+"/user_details.pkl", "rb") as input_file:
	user_details = pickle.load(input_file)

with open(r"Results/"+entry+"/post_test_responses.pkl", "rb") as input_file:
	question_responses = pickle.load(input_file)


print(user_details)
print(question_responses)