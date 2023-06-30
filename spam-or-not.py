import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

path = 'C:/Users/SIRIKONDA GANESH/Desktop/Task_2/spambase_csv.csv'
df = pd.read_csv(path)

X = df.iloc[:,:-1].values
y = df.iloc[:,57].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.23, random_state = 0)



from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)


pred = logreg.predict(x_test)

my_submission = pd.DataFrame({'spam_or_not(if spam value=1 or else if not spam value =0)': pred})

my_submission.to_csv('submission.csv', index=False)


my_submission