import pickle

# ----------------- #
# load model 
# ----------------- #
def load_model(path: str):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model 