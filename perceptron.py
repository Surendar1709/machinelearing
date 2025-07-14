import  numpy as np


print("______basic or gate function_________________")
x=np.array([[0,0],[0,1],[1,0],[1,1]])  # input
y=np.array([0,1,1,1])  # expcted output
w=np.array([1,1])  # weight 
b=-0.5          #  bias  values 
print("input",x)
print("Expeted output",y)
print("weight",w)
print("bias",b)



def activation(z):

    if z>0:
        return 1
    else:
        return 0
predict=[]
for a in x:
    z=np.dot(a,w)+b
    predict.append(activation(z))

print(predict)


print()
print()
print()

epochs=100
alpha=0.2
w0=np.random.random()
w1=np.random.random()
w2=np.random.random()


print("____initial_weight______")
print(f"w0{w0}  w1{w1}  w2{w2}")
del_w0=1

del_w1=1

del_w2=1

# train data
train_data=np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
op=np.array([0,1,1,1,1,1,1,1])

bias=0
for i in range(epochs):
    j=0
    for x in train_data:
        y_hat=w0*x[0] +w1*x[1] +w2*x[2]+bias

        if y_hat >0:
            act=1
        else:
            act=0

        err=op[j]-act


        del_w0=alpha*x[0]*err
        del_w1=alpha*x[1]*err
        del_w2=alpha*x[2]*err

        w0=w0+del_w0
        w1=w1+del_w1
        w2=w2+del_w2

        j=j+1
        print("Epoces",i+1,"error=",err)
        print(del_w0,del_w1,del_w2)
       
print("  Fianl  weight --")
print("w0=",w0,"w1=",w1,'w2=',w2)



# four types of  classfication 
'''
1. binary Classifiction
2.mutlticlass  Classification 
3.multilabel    Classification 
4 imbalance   Classification  


Leaner type  Terminology 

1 lazy  learner  spent more time  on   predicting  and less time  traning  Knn

2 eager  learners     spent more  time  traning  and   less time  on  predicting 



sklit learn classfiction algorithms 

1 logestic  Regression   focus   use  cases al  yes or no   true  or  flase  o or 1 
2  knn    algorithm  assumes    that  simailar  things   exists  in  close   proximity 
3  dicision tree algorithm    used to slove the  regression and classification problems 
4 Random  forest   is a classifer that    conatain  a number   of decision   trees on variour  
 subsets of the  given dataset and takes  the average  to improve   the  predictive  accuracy  
 of that dataset 

5  supprot vector  Machine      the svm    algorithm  create   the best  line or decision  boundary 
that  can  segregate   n-dimensional   space   into clasess  

6  Naive   Bayes classfier    assuemes  that   the effect   of  a partcular feature   in  a class   independent 
   other   Features 
'''

