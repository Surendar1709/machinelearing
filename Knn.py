

#  comes  under  supervied machine learing    classification  algorithm knn
#  Similar  things  are   extsting  in close  proximaty 
#  totalyy based on  feature  similarity  

#  parameter  tunning   is  plays the  important   role   in  knn
#  how to  choose the k  value  
# 

'''  K  value   is choosen by  applying the  following  method 
       1 .hit  and  trial 
       2 .sqrt(n)  where   n stands  for total   number  of data samples in dataset 
       3  odd value   of k  is  selected   to avoid  confusion   between  two  classes  of data


       
    where      can  we  use  Knn    algorithms  
        1.Data set  shoud be  properly labeled  
        2. Data shoud be  noise  free 
        3  works  very  well  on  small  scale Datasets 
        4  Knn is better   when  yo   want   to create   models with  higher   accuracy  on cost  of 
           computational  resources  


'''  
#  how  does  knn Works  

'''   Euclidean  Distance   is the  distance  between two  point the plan 
  step 1 :memorize the   label  Data 
  step 2 choose the k   as  no of  neigbhour  
  step 3 compute the  distance 
  step 4  find the  nearest  neighbor  
  step 5  predict  by  voting  and  averaging 
  step 6 optimiing  k  '''

import  numpy  as  np
import  pandas as pd 
import  matplotlib.pyplot as plt 
from sklearn import datasets
import  seaborn as sns


data=datasets.load_wine(as_frame=True )
print(data)


# seprating    Feature  data  and     targeting  data(input and  excepeted output )  


print()
print()
X=data.data
Y=data.target
names=data.target_names
print(names)


# converting into the  dataFrame 

df=pd.DataFrame(X,columns=data.feature_names)

df['Wine class']=data.target
df['Wine class'] = df['Wine class'].replace(
    to_replace=[0, 1, 2],
    value=['class_0', 'class_1', 'class_2']
)

print(df)

sns.pairplot(data=df,hue='Wine class',palette='Set2')
plt.show()