import pickle
with open('post_test_responses.pkl', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
    print(content)
