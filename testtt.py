import pickle

params = pickle.load(open("params.pkl", "rb"))
print(params[1].policy["params"])