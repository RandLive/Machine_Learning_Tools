# The following code illustrates how TPOT can be employed for performing a regression task over the Boston housing prices dataset.
# url: http://rhiever.github.io/tpot/examples/

from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

housing = load_boston()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')

# Running this code should discover a pipeline (exported as tpot_boston_pipeline.py) that achieves at least 10 mean squared error (MSE) on the test set:

import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1),
                     tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = GradientBoostingRegressor(alpha=0.85, learning_rate=0.1, loss="ls",
                                              max_features=0.9, min_samples_leaf=5,
                                              min_samples_split=6)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

