import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

house_data = pd.read_csv('house.csv')

sample = np.random.randint(house_data.size, size=house_data.size*0.1)

plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)

# # GET THETA VALUE (result:
# [[-283.37836117]
#  [  40.97116431]]
#


X = np.matrix([np.ones(house_data.shape[0]), house_data['surface'].values]).T
y = np.matrix(house_data['loyer']).T

xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8)

theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


plt.plot([0, 250],
         [theta.item(0), theta.item(0) + 250 * theta.item(1)],
         linestyle="--",
         c="#000000")

plt.show()

print("Pour une surface de 122m², le loyer devrait être approximativement de ", theta.item(0) + theta.item(1) * 122, "€.")
