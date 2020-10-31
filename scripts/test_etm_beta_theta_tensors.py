import torch.tensor
import joblib
import numpy as np

tensor_dict = joblib.load("TENSORS_FILE") 
print(tensor_dict["beta"]) # tensor obtido a partir de model.get_beta() no ETM
print(tensor_dict["beta"].size()) # dimensoes do tensor parecem ser KxV (topicos x vocabulario)
beta_sum = torch.sum(tensor_dict["beta"], 1)
print(beta_sum) # linhas somam 1, mostrando que estao normalizadas
print(beta_sum.size())

array = tensor_dict["beta"].numpy()
filter_fun = array < 0
print(array[filter_fun]) # o tensor nao possui elementos negativos

print("*"*20)

tensor_dict = joblib.load("TENSORS_FILE") 
print(tensor_dict["theta"]) # tensor obtido a partir de model.get_theta() no ETM
print(tensor_dict["theta"].size()) # dimensoes do tensor parecem ser DxK (documentos x topicos)
theta_sum = torch.sum(tensor_dict["theta"], 1)
print(theta_sum) # linhas somam 1, mostrando que estao normalizadas
print(theta_sum.size())

array = tensor_dict["theta"].numpy()
filter_fun = array < 0
print(array[filter_fun]) # o tensor nao possui elementos negativos
