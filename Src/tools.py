import torch as T

def save_model(model, path):
    T.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(T.load(path))
    
    return model
