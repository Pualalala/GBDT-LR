# -*- coding: utf-8 -*-
import xlrd    #xlrd是读excel
import numpy
import time
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import LeaveOneOut
import math
import numpy.linalg as LA
from sklearn.cluster import KMeans
import random
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection,metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
startTime = time.time()
a=open(r'.\data\Known disease-miRNA association number.xlsx')
b=open(r'.\data\Disease semantic similarity matrix 1.xlsx')
c=open(r'.\data\Disease semantic similarity matrix 2.xlsx')
d=open(r'.\data\Disease semantic similarity weighting matrix.xlsx')
e=open(r'.\data\miRNA functional similarity matrix.xlsx')
f=open(r'.\data\miRNA functional similarity weighting matrix.xlsx')
g=open(r'.\data\miRNA number.xlsx')
h=open(r'.\data\disease number.xlsx')
SS=numpy.zeros((383,383))  #383*383阶的零矩阵
A=numpy.zeros((383,495))
KD=numpy.zeros((383,383))
SD=numpy.zeros((383,383))
DWM=numpy.zeros((383,383))
MWM=numpy.zeros((495,495))
FS=numpy.zeros((495,495))
SM=numpy.zeros((495,495))
KM=numpy.zeros((495,495))
D1=numpy.zeros(10950)   #生成包含10950个元素的零矩阵  array([ 0., 0., 0., 0., 0.])
D2=numpy.zeros(10950)
D3=numpy.zeros(10950)
D4=numpy.zeros(10950)
D5=numpy.zeros(10950)
D6=numpy.zeros(10950)
D7=numpy.zeros(10950)
D8=numpy.zeros(10950)
D9=numpy.zeros(10950)
D10=numpy.zeros(10950)
D11=numpy.zeros(10950)
D12=numpy.zeros(10950)
D13=numpy.zeros(10950)
D14=numpy.zeros(10950)
D15=numpy.zeros(10950)
D16=numpy.zeros(10950)
D17=numpy.zeros(10950)
D18=numpy.zeros(10950)
D19=numpy.zeros(10950)
xlsx1=xlrd.open_workbook(r'.\data\Disease semantic similarity matrix-model 1.xlsx')
xlsx2=xlrd.open_workbook(r'.\data\Disease semantic similarity matrix-model 2.xlsx')  #打开Excel文件读取数据
sheet1=xlsx1.sheets()[0]        ##通过索引顺序获取
sheet2=xlsx2.sheets()[0]
for i in range(383):
    for j in range(383):
        s1=sheet1.row_values(i)  #获取整行和整列的值（数组）
        s2=sheet2.row_values(i)
        m=s1[j]
        n=s2[j]
        SS[i,j]=float(m+n)/2              #综合疾病语义相似性Obtain disease semantic similarity SS
xlsx3=xlrd.open_workbook(r'.\data\Known disease-miRNA association number.xlsx')
sheet3=xlsx3.sheets()[0]
for i in range(5430):
    s3=sheet3.row_values(i)
    m=int(s3[0])
    n=int(s3[1])
    A[n-1,m-1]=1                    #Obtain adjacency matrix A   邻接矩阵  行：疾病 列：miRNA
xlsx4=xlrd.open_workbook(r'.\data\Disease semantic similarity weighting matrix.xlsx')
sheet4=xlsx4.sheets()[0]
for i in range(383):
    for j in range(383):
        s4=sheet4.row_values(i)
        DWM[i,j]=s4[j]               #Get disease semantic weighting matrix DJQ
xlsx5=xlrd.open_workbook(r'.\data\miRNA functional similarity weighting matrix.xlsx')
sheet5=xlsx5.sheets()[0]
for i in range(495):
    for j in range(495):
        s5=sheet5.row_values(i)
        MWM[i,j]=s5[j]               #Get miRNA functional similarity weighting matrix MJQ
xlsx6=xlrd.open_workbook(r'.\data\miRNA functional similarity matrix.xlsx')
sheet6=xlsx6.sheets()[0]
for i in range(495):
    for j in range(495):
        s6=sheet6.row_values(i)
        FS[i,j]=s6[j]                  #Get miRNA functional similarity matrix FS
C=np.asmatrix(A)           #矩阵C   asmatrix(data[, dtype])：将输入解释为矩阵.
gamd=383/(LA.norm(C,'fro')**2);   #norm(X,'fro') 计算矩阵 (向量) C的F范数;F范数是把一个矩阵中每个元素的平方求和后开根号；**两个乘号就是乘方，比如2**4,结果就是2的4次方，结果是16
kd=np.mat(np.zeros((383,383)))     #创建一个零矩阵
km=np.mat(np.zeros((495,495)))
D=C*C.T;          #C*C的转置矩阵
for i in range(383):
        for j in range(i,383):
            kd[j,i]=np.exp(-gamd*(D[i,i]+D[j,j]-2*D[i,j]))   # 高斯核相似计算
kd=kd+kd.T-np.diag(np.diag(kd))   #两次使用diag() 获得某二维矩阵的对角矩阵：
KD=np.asarray(kd)                  #Obtain Gaussian interaction profile kernel similarity for disease SD
kd=[]
SD = np.multiply(SS,DWM)+np.multiply(KD,(1-DWM))     #np.multiply(A,B)矩阵对应元素位置相乘   疾病综合相似性
SD=np.asarray(SD)                                   #DWM  is disease semantic weighting matrix
gamam = 495/(LA.norm(C,'fro')**2);
E=C.T*C;
for i in range(495):
    for j in range(i,495):
        km[i,j]=np.exp(-gamam*(E[i,i]+E[j,j]-2*E[i,j]))
km=km+km.T-np.diag(np.diag(km))
KM=np.asarray(km)
km=[]
SM = np.multiply(FS,MWM)+np.multiply(KM,(1-MWM))
SM=np.asarray(SM)                                         #Obtain Gaussian interaction profile kernel similarity for miRNA SM
#K_mean clustering
unknown=[]
known=[]
for x in range(383):
    for y in range(495):
        if A[x,y]==0:
            unknown.append((x,y))    #append() 方法向列表的尾部添加一个新的元素
        else:
            known.append((x,y))               #Divide the samples into Major and Minor
major=[]
for z in range(184155):
        q=SD[unknown[z][0],:].tolist()+SM[unknown[z][1],:].tolist()#tolist()将数组或者矩阵转换成列表    X[1,:] 取第一行的所有列数据
        major.append(q)
kmeans=KMeans(n_clusters=23, random_state=0).fit(major) # n_clusters 和random_state。其中，前者表示你打算聚类的数目，默认情况下是8。后者表示产生随机数的方法
center=kmeans.cluster_centers_      # 获取簇心 #获得训练后模型23类中心点位置坐标 为23x2的矩阵 fit()方法是对Kmeans确定类别以后的数据集进行聚类
center_x=[]
center_y=[]
for j in range(len(center)):
    center_x.append(center[j][0])
    center_y.append(center[j][1])
labels=kmeans.labels_   #标注每个点的聚类结果  得到聚类后每一组数据对应的类型标签，数据为一个列表labels[1,2,3,1......]对应每一组数据的类别
type1_x=[]
type1_y=[]
type2_x=[]
type2_y=[]
type3_x=[]
type3_y=[]
type4_x=[]
type4_y=[]
type5_x=[]
type5_y=[]
type6_x=[]
type6_y=[]
type7_x=[]
type7_y=[]
type8_x=[]
type8_y=[]
type9_x=[]
type9_y=[]
type10_x=[]
type10_y=[]
type11_x=[]
type11_y=[]
type12_x=[]
type12_y=[]
type13_x=[]
type13_y=[]
type14_x=[]
type14_y=[]
type15_x=[]
type15_y=[]
type16_x=[]
type16_y=[]
type17_x=[]
type17_y=[]
type18_x=[]
type18_y=[]
type19_x=[]
type19_y=[]
type20_x=[]
type20_y=[]
type21_x=[]
type21_y=[]
type22_x=[]
type22_y=[]
type23_x=[]
type23_y=[]
for i in range(len(labels)):    #将所有未知关联（疾病，miRNA序列号）进行分类，分成23类并记录
    if labels[i]==0:
        type1_x.append(unknown[i][0])
        type1_y.append(unknown[i][1])
    if labels[i]==1:
        type2_x.append(unknown[i][0])
        type2_y.append(unknown[i][1])
    if labels[i]==2:
        type3_x.append(unknown[i][0])
        type3_y.append(unknown[i][1])
    if labels[i]==3:
        type4_x.append(unknown[i][0])
        type4_y.append(unknown[i][1])
    if labels[i]==4:
        type5_x.append(unknown[i][0])
        type5_y.append(unknown[i][1])
    if labels[i]==5:
        type6_x.append(unknown[i][0])
        type6_y.append(unknown[i][1])
    if labels[i]==6:
        type7_x.append(unknown[i][0])
        type7_y.append(unknown[i][1])
    if labels[i]==7:
        type8_x.append(unknown[i][0])
        type8_y.append(unknown[i][1])
    if labels[i]==8:
        type9_x.append(unknown[i][0])
        type9_y.append(unknown[i][1])
    if labels[i]==9:
        type10_x.append(unknown[i][0])
        type10_y.append(unknown[i][1])
    if labels[i]==10:
        type11_x.append(unknown[i][0])
        type11_y.append(unknown[i][1])
    if labels[i]==11:
        type12_x.append(unknown[i][0])
        type12_y.append(unknown[i][1])
    if labels[i]==12:
        type13_x.append(unknown[i][0])
        type13_y.append(unknown[i][1])
    if labels[i]==13:
        type14_x.append(unknown[i][0])
        type14_y.append(unknown[i][1])
    if labels[i]==14:
        type15_x.append(unknown[i][0])
        type15_y.append(unknown[i][1])
    if labels[i]==15:
        type16_x.append(unknown[i][0])
        type16_y.append(unknown[i][1])
    if labels[i]==16:
        type17_x.append(unknown[i][0])
        type17_y.append(unknown[i][1])
    if labels[i]==17:
        type18_x.append(unknown[i][0])
        type18_y.append(unknown[i][1])
    if labels[i]==18:
        type19_x.append(unknown[i][0])
        type19_y.append(unknown[i][1])
    if labels[i]==19:
        type20_x.append(unknown[i][0])
        type20_y.append(unknown[i][1])
    if labels[i]==20:
        type21_x.append(unknown[i][0])
        type21_y.append(unknown[i][1])
    if labels[i]==21:
        type22_x.append(unknown[i][0])
        type22_y.append(unknown[i][1])
    if labels[i]==22:
        type23_x.append(unknown[i][0])
        type23_y.append(unknown[i][1])
type=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]                  #23簇
mtype=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
dataSet=[]
mtype1=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
print('Completed.Took %f s.' % (time.time() - startTime))
for k1 in range(len(type1_x)):
    type[0].append((type1_x[k1],type1_y[k1]))
for k2 in range(len(type2_x)):
    type[1].append((type2_x[k2],type2_y[k2]))
for k3 in range(len(type3_x)):
    type[2].append((type3_x[k3],type3_y[k3]))
for k4 in range(len(type4_x)):
    type[3].append((type4_x[k4],type4_y[k4]))
for k5 in range(len(type5_x)):
    type[4].append((type5_x[k5],type5_y[k5]))
for k6 in range(len(type6_x)):
    type[5].append((type6_x[k6],type6_y[k6]))
for k7 in range(len(type7_x)):
    type[6].append((type7_x[k7],type7_y[k7]))
for k8 in range(len(type8_x)):
    type[7].append((type8_x[k8],type8_y[k8]))
for k9 in range(len(type9_x)):
    type[8].append((type9_x[k9],type9_y[k9]))
for k10 in range(len(type10_x)):
    type[9].append((type10_x[k10],type10_y[k10]))
for k11 in range(len(type11_x)):
    type[10].append((type11_x[k11],type11_y[k11]))
for k12 in range(len(type12_x)):
    type[11].append((type12_x[k12],type12_y[k12]))
for k13 in range(len(type13_x)):
    type[12].append((type13_x[k13],type13_y[k13]))
for k14 in range(len(type14_x)):
    type[13].append((type14_x[k14],type14_y[k14]))
for k15 in range(len(type15_x)):
    type[14].append((type15_x[k15],type15_y[k15]))
for k16 in range(len(type16_x)):
    type[15].append((type16_x[k16],type16_y[k16]))
for k17 in range(len(type17_x)):
    type[16].append((type17_x[k17],type17_y[k17]))
for k18 in range(len(type18_x)):
    type[17].append((type18_x[k18],type18_y[k18]))
for k19 in range(len(type19_x)):
    type[18].append((type19_x[k19],type19_y[k19]))
for k20 in range(len(type20_x)):
    type[19].append((type20_x[k20],type20_y[k20]))
for k21 in range(len(type21_x)):
    type[20].append((type21_x[k21],type21_y[k21]))
for k22 in range(len(type22_x)):
    type[21].append((type22_x[k22],type22_y[k22]))
for k23 in range(len(type23_x)):
    type[22].append((type23_x[k23],type23_y[k23]))                 #Divide Major into 23 clusters by K-means clustering

for t in range(23):
    mtype1[t]=random.sample(type[t],int((len(type[t])/len(labels))*5430))
for m2 in range(383):
    for n2 in range(495):
        for z2 in range(23):
            if (m2,n2) in mtype1[z2]:
                dataSet.append((m2,n2))                         #Store the randomly extracted 23X240 samples in the dataSet
for m3 in range(383):
    for n3 in range(495):
        if A[m3,n3]==1:                          #dataset存的是（疾病号，mirna号）
            dataSet.append((m3,n3))                              #Combine Major and Minor into training samples containing 10,950 samples
#Decision tree
sumy1=numpy.zeros(10950)
sumy2=numpy.zeros(10950)
sumy3=numpy.zeros(10950)
sumy4=numpy.zeros(10950)
sumy5=numpy.zeros(10950)
sumy6=numpy.zeros(10950)
sumy7=numpy.zeros(10950)
sumy8=numpy.zeros(10950)
sumy9=numpy.zeros(10950)
sumy10=numpy.zeros(10950)
sumy11=numpy.zeros(10950)
sumy12=numpy.zeros(10950)
sumy13=numpy.zeros(10950)
sumy14=numpy.zeros(10950)
sumy15=numpy.zeros(10950)
sumy16=numpy.zeros(10950)
sumy17=numpy.zeros(10950)
sumy18=numpy.zeros(10950)
sumy19=numpy.zeros(10950)
sumy20=numpy.zeros(10950)
x=[]
x1=[]
x2=[]
y=[]
D=numpy.ones(10950)*1.0/10950.0         #Initialize the weight of the training sample
for xx in dataSet:            #for example in dataset:for循环，在数据dataset（可以是列表、元组、集合、字典等）中 逐个取值存入 变量 example中，然后运行循环体
    q=SD[xx[0],:].tolist()+SM[xx[1],:].tolist()    #dataset存的是（疾病号，mirna号）  SD疾病综合相似性矩阵   X[1,:] 取第一行的所有列数据
    x.append(q)
    if (xx[0],xx[1]) in known:
        y.append(1)    #标签  正样本为1
    else:
        y.append(0)
ys=numpy.array(y)
xs=numpy.array(x)


"""
GBDT=GradientBoostingClassifier(n_estimators = 12,max_depth=5,min_samples_leaf=13)
# 训练模型
GBDT.fit(xs,ys)
OHE = OneHotEncoder()
OHE.fit(GBDT.apply(xs)[:, :, 0])#model.apply(X_train)返回训练数据X_train在训练好的模型里每棵树中所处的叶子节点的位置（索引）
LR = LogisticRegression()
tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
for train,test in cv.split(xs):
    #通过训练数据，使用svm线性核建立模型，并对测试集进行测试，求出预测得分
    #probas_ = classifier.fit(xs[train],ys[train]).predict_proba(xs[test])
    LR.fit(OHE.transform(GBDT.apply(xs[train])[:, :, 0]),ys[train])
    probas_ = LR.predict_proba(OHE.transform(GBDT.apply(xs[test])[:, :, 0]))
    # Compute ROC curve and area the curve
    #通过roc_curve()函数，求出fpr和tpr，以及阈值
    fpr, tpr, thresholds = roc_curve(ys[test], probas_[:, 1])
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)#auc(fpr,tpr)，其返回值即为AUC的值。
    aucs.append(roc_auc)

"""


GBDT=GradientBoostingClassifier(n_estimators = 12,max_depth=5,min_samples_leaf=13)
# 训练模型
GBDT.fit(xs,ys)
OHE = OneHotEncoder()
OHE.fit(GBDT.apply(xs)[:, :, 0])#model.apply(X_train)返回训练数据X_train在训练好的模型里每棵树中所处的叶子节点的位置（索引）
LR = LogisticRegression()
LR.fit(OHE.transform(GBDT.apply(xs)[:, :, 0]),ys)


"""
gbm0 = GradientBoostingClassifier(n_estimators = 12,max_depth=5,min_samples_leaf=13)    #range(0, 30, 5)  # 从 0 开始到 30步长为 5
gbm0.fit(x,y)
y_pred = gbm0.predict(x)# 返回预测标签
y_predprob = gbm0.predict_proba(x)[:,1] #predict_proba# 返回预测属于某标签的概率并且每一行的概率和为1。
#print("Accuracy : %.4g" ,metrics.accuracy_score(y.values, y_pred))
#print ("AUC Score (Train): %f" ,metrics.roc_auc_score(y, y_predprob))
#print('Completed.Took %f s.' % (time.time() - startTime))
#print ("Accuracy :" ,metrics.accuracy_score(ys, y_pred))
#print ("AUC Score (Train):", metrics.roc_auc_score(ys, y_predprob))
#cv =LeaveOneOut(len(ys))

"""
for yy in unknown:
    q1=SD[yy[0],:].tolist()+SM[yy[1],:].tolist()
    x1.append(q1)

#fs=gbm0.predict_proba(x1)
fs=LR.predict_proba(OHE.transform(GBDT.apply(x1)[:, :, 0]))
px1=fs[:,1].tolist()
xlsx7=xlrd.open_workbook(r'.\data\disease number.xlsx')
xlsx8=xlrd.open_workbook(r'.\data\miRNA number.xlsx')
sheet7=xlsx7.sheets()[0]
sheet8=xlsx8.sheets()[0]
px1=numpy.matrix(px1)
Sampleindex=numpy.argsort(-px1).tolist()   #.argsort(-px1)返回将对数组排序的索引   从大到小
Sampleindex=Sampleindex[0]
f=open(r'Prediction results final.txt','a+')
f.writelines(['disease','\t','miRNA','\t','Score','\n'])
f.close()
for i in range(184155):
    a=fs[:,1][Sampleindex[i]]
    s7=sheet7.row_values(unknown[Sampleindex[i]][0])
    s8=sheet8.row_values(unknown[Sampleindex[i]][1])
    f=open(r'Prediction results final.txt','a+')
    f.writelines([s7[1],'\t',s8[1],'\t',str(a),'\n'])
    f.close()                                        #Obtain the prediction results for all unknown samples
print('Completed.Took %f s.' % (time.time() - startTime))
