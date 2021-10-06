# Adam Rilatt
# 10 / 14 / 20
# Tic Tac Toe Bot -- Logistic Regression Model

'''
This script analyzes data generated from the UTTT board to find statistically
significant features or otherwise ways to increase accuracy. In short, it's a
test ground.
'''

import h5py
import numpy as np
import statsmodels.api as sm
from itertools import compress
from sklearn   import metrics
from sklearn.feature_selection import RFE
from sklearn.linear_model      import LogisticRegression
from sklearn.model_selection   import train_test_split

# user-choice parameters
NUM_FEATURE = 27
TEST_TRAIN  = 0.2
SEED        = 42
RECORD_FILE = h5py.File('tictac_record_test.h5', 'r')

np.random.seed(SEED)

# for this model, we predict the chance that X will win
y = [row[0] for row in RECORD_FILE['Y'][:]]
x = RECORD_FILE['X'][:len(y)]

logreg = LogisticRegression()

# recursive feature elimination
print("Eliminating all but %d features..." % NUM_FEATURE)
rfe = RFE(logreg, n_features_to_select = NUM_FEATURE)
rfe = rfe.fit(x, y)
rfe_x = [list(compress(row, rfe.support_)) for row in x]

# testing p-vals for efficient parameterizations -- how important are
# these parameters?
#logit_model = sm.Logit(y, rfe_x)
#result = logit_model.fit()
#print(result.summary2())


print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(rfe_x, y, test_size = TEST_TRAIN,
                                                           random_state = SEED)
logreg = LogisticRegression(solver = 'lbfgs')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
    logreg.score(X_test, y_test)
))

cm = metrics.confusion_matrix(y_test, prediy_pred)
print(cm)
