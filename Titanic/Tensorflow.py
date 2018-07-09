import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

# Style
sns.set_palette("deep", 8)

# Load Data
data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Drop whats unecessary
data = data.drop(columns=['Cabin', 'Ticket'])
test_data = test_data.drop(columns=['Cabin', 'Ticket'])

# ----------------------
# Add feature FamilySize
# ----------------------
data['FamilySize'] = [x + y for x, y in zip(data['Parch'], data['SibSp'])]
test_data['FamilySize'] = [x + y for x, y in zip(test_data['Parch'], test_data['SibSp'])]

# Categorize
def family_size_category(size):
    if size >= 5: return 3
    elif 3 <= size < 5: return 2
    elif 1 <= size < 3: return 1
    else: return 0

data["FamilySize"] = data['FamilySize'].map(family_size_category)
test_data["FamilySize"] = test_data['FamilySize'].map(family_size_category)


# ---------------------
# Convert Sex to binary
# ---------------------
data['Sex'] = data['Sex'].map({'female': 1, 'male': 0}).astype(int)
test_data['Sex'] = test_data['Sex'].map({'female': 1, 'male': 0}).astype(int)


# ------------------
# Deal with Embarked
# ------------------
# Fill Nan - Embarked
# Find most frequent category in column (mode()[0]) an fill NaN with it
embarked_freq = data['Embarked'].dropna().mode()[0]
data['Embarked'] = data['Embarked'].fillna(embarked_freq)

# Convert to numeric
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)
test_data['Embarked'] = test_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)


# --------------------
# Get titles from Name
# --------------------
# Use .extract(REGEX) to extract titles only
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


for i, row in data.iterrows():
    if (row['Title'] not in ["Mr", "Miss", "Mrs", "Master"]):
        data['Title'][i] = "Other"

for i, row in test_data.iterrows():
    if (row['Title'] not in ["Mr", "Miss", "Mrs", "Master"]):
        test_data['Title'][i] = "Other"

# Convert to int
data['Title'] = data['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Other': 4}).astype(int)
test_data['Title'] = test_data['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Other': 4}).astype(int)


# ---------------
# Fare Categories
# ---------------
# Fill missing fare on test data
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())

# Apply categories with labels
data['FareCategory'] = pd.qcut(data['Fare'], 4, labels=[0,1,2,3]).astype(int)
test_data['FareCategory'] = pd.qcut(test_data['Fare'], 4, labels=[0,1,2,3]).astype(int)


# --------
# Clean up
# --------
# Save PassengerId for later
passenger_id = test_data['PassengerId']

# Drop whats unecessary
data = data.drop(columns=['SibSp', 'Parch', 'PassengerId', 'Fare', 'Name'])
test_data = test_data.drop(columns=['SibSp', 'Parch', 'PassengerId', 'Fare', 'Name'])


# --------------------------------------------------------
# Deal with Age NaN and divide into categories (Train set)
# --------------------------------------------------------
#Replace NaN with mean for Age
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Select training data
XClean = data.dropna(axis=0, how='any')
X = XClean[['Title', 'Pclass', 'FareCategory', 'Embarked']]
y = data[['Age']].dropna()

# Select test data (rows missing Age)
XTestClean = data.loc[pd.isnull(data[['Age']]).any(axis=1)]
XTest = XTestClean[['Title', 'Pclass', 'FareCategory', 'Embarked']]
yToFill = XTestClean[['Age']]

# Fit and predict
linreg = LinearRegression()
linreg.fit(X, y)

# predict
yPred = linreg.predict(XTest).flatten().tolist()

# Replace NaN with predicted Age values
yToFill['Age'] = yPred
data['Age'] = data.combine_first(yToFill)

# Categorize
def age_category(age):
    if age >= 60: return 4
    elif 21 <= age < 60: return 3
    elif 12 <= age < 21: return 2
    elif 3 <= age < 12: return 1
    else: return 0

data["Age"] = data['Age'].map(age_category)

# Deal with Age NaN and divide into categories (Test set)

# Select data (rows missing Age)
XTestClean = test_data.loc[pd.isnull(test_data[['Age']]).any(axis=1)]
XTest = XTestClean[['Title', 'Pclass', 'FareCategory', 'Embarked']]
test_yToFill = XTestClean[['Age']]

# predict
yPred = linreg.predict(XTest).flatten().tolist()

# Replace NaN with predicted Age values
test_yToFill['Age'] = yPred
test_data['Age'] = test_data.combine_first(test_yToFill)

# Categorize
test_data["Age"] = test_data['Age'].map(age_category)

# ---------
# Visualize
# ---------
data.head()

# Create figure & adjust
fig, (ax1, ax2,ax3) = plt.subplots(3, 3, gridspec_kw={'height_ratios':[1,1,1]}, figsize=(16,11), sharey=True)
fig.subplots_adjust(wspace=0.1)

# Plot
sns.barplot(x="Pclass", y="Survived", data=data, ax=ax1[0])
sns.barplot(x="Sex", y="Survived", data=data, ax=ax1[1])
sns.barplot(x="FamilySize", y="Survived", data=data, ax=ax1[2])
sns.barplot(x="Embarked", y="Survived", data=data, ax=ax2[0])
sns.barplot(x="Age", y="Survived", data=data, ax=ax2[1])
sns.barplot(x="Title", y="Survived", data=data, ax=ax2[2])
sns.barplot(x="FareCategory", y="Survived", data=data, ax=ax3[1])

# Create dummy variables for categorical that don't have a linear slope of relationship to survivability variables and delete one to avoid trap
#---------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
data = pd.get_dummies(data, columns=['Embarked', 'Title', 'FamilySize'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Embarked', 'Title', 'FamilySize'], drop_first=True)


# Separate data
X_train, X_test, y_train, y_test = train_test_split(data.drop('Survived', axis=1),
                                                    data.loc[:, 'Survived'],
                                                    test_size=0.3,
                                                    random_state=101)
X_train = np.array(X_train).astype('float32')
y_train = np.array(y_train).reshape(len(y_train),1).astype('float32')
X_test = np.array(X_test).astype('float32')
y_test = np.array(y_test).reshape(len(y_test),1).astype('float32')

test_X = test_data

# ----------
# TENSORFLOW
# ----------
n_hidden1 = len(X_train[0])
n_hidden2 = 6
n_classes = 1
epochs = 1000

# Avoid errors
tf.reset_default_graph()

# PLACEHOLDERS
X = tf.placeholder(tf.float32, [None, n_hidden1])
y_true = tf.placeholder(tf.float32, [None, n_classes])
step = tf.placeholder(tf.float32)

# VARIABLES
W1 = tf.Variable(tf.zeros([n_hidden1, n_hidden2]))
B1 = tf.Variable(tf.zeros([n_hidden2]))
W2 = tf.Variable(tf.zeros([n_hidden2, n_classes]))
B2 = tf.Variable(tf.zeros([n_classes]))

# MODEL
Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)
Y = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)


# COST and OPTIMIZER
learning_rate = 0.01 * (0.998**step)
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y,labels=y_true))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# SESSION
init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./output",sess.graph)
    sess.run(init)

    for e in range(epochs):
        print("-------------")
        print(f"Epoch {e}")
        sess.run(optimizer, feed_dict={X:X_train, y_true: y_train, step:e})
        print(f'Cross-entropy: ', cross_entropy.eval(feed_dict={X:X_train, y_true: y_train, step:e}))
        print(f'Learning rate: ', learning_rate.eval(feed_dict={X:X_train, y_true: y_train, step:e}))

        # Test the model
        matches = tf.cast(tf.equal(tf.round(Y),y_true),tf.float32)
        acc = tf.reduce_mean(matches)
        print(sess.run(acc,feed_dict={X:X_test, y_true:y_test, step:e}))
    writer.close()
