import pandas as pd
import numpy as np

 
data = pd.read_excel(r"/content/Lab Session Data.xlsx", sheet_name="Purchase data")

 
A = data.loc[:, ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
C = data[['Payment (Rs)']].values
 
print(f"A = {A}")
print(f"C = {C}")

dimensionality = A.shape[1]
print(f"the dimensionality of the vector space is  = {dimensionality}")

vector_space = len(A)
print(f"the number of vectors in the vector space is = {vector_space}")

rank_A = np.linalg.matrix_rank(A)
print(f"the rank of the matrix A is = {rank_A}")

pseudo_inverse = np.linalg.pinv(A)
print(f"the pseudo inverse of matrix A is = {pseudo_inverse}")

cost = np.linalg.lstsq(A, C, rcond=None)[0]
print(f"the cost of each product that is available for sale is  = {cost.flatten()}")
