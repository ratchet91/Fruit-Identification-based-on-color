import pandas as pa
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data="fruit_data_with_colors.txt"
name=['mass','width','height','color_score']
fruit_data=pa.read_table(data)
x=fruit_data[name]
y=fruit_data['fruit_label']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

model=DecisionTreeClassifier()
model.fit(x,y)
opt_label=model.predict([[180,7,10,0.59]])
print opt_label

label=['apple','mandarin','orange','lemon']
c=opt_label[0]
print label[c-1]

