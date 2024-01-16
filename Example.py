# %%
from CGA_tree import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, auc, accuracy_score
from HDlogistic.hdlogistic import *
from numpy import random
# %%
### basic function
def ma_cov(p,ar = 0.5):
    cov = [1. for _ in range(p)]
    cov = np.diag(cov)
    for i in range(p-1):
        cov[i,i+1] = ar
        cov[i+1,i] = ar
    return cov

def ar_cov(p,ar = 0.5):
    cov = [[0. for _ in range(p)] for _ in range(p)]
    for i in range(p):
        for j in range(p-i):
            cov[j][j+i] = ar**i
            cov[j+i][j] = ar**i
    return cov
def poly_cov(p,ar = 0.5):
    cov = [[0. for _ in range(p)] for _ in range(p)]
    for i in range(p):
        for j in range(p-i):
            cov[j][j+i] = 1/(i+2)
            cov[j+i][j] = 1/(i+2)
    return cov
def sure_screen(idx,use_idx):
    sure_screen = [i in idx for i in use_idx]
    n_contain = sum(sure_screen)
    var_len = len(idx)
    cover = 0; e = 0; e1 = 0; e2 = 0; e3 = 0
    if all(sure_screen):
        cover = 1
        if var_len == len(use_idx):
            e = 1
        elif var_len == (len(use_idx)+1):
            e1 = 1
        elif var_len == (len(use_idx)+2):
            e2 = 1
        elif var_len == (len(use_idx)+3):
            e3 = 1
    return var_len,n_contain,cover,e,e1,e2,e3
# %%
def generate_data1(seed,n,p):
    random.seed(seed)
    cov = ar_cov(p,0.7)
    x = random.multivariate_normal([0 for _ in range(p)],cov,size = n)
    x = random.normal(0,1,size = (n,p))
    x = np.concatenate(([[1] for i in range(n)],x),axis = 1)
    beta = [1,2,2,3,4,2]+[0 for i in range(p-5)]
    PI = 1/(1+np.exp(-(x@beta)))
    y = random.binomial(1,np.reshape(PI,n),n)
    x = pd.DataFrame(x)
    x.columns = ['V'+str(i) for i in range((x.shape[1]))]
    x = x.iloc[:,1:]

    y=  pd.DataFrame(y)
    use_idx = [0,1,2,3,4]
    return x,y,PI,beta,use_idx

def generate_data2(seed,n,p):
    random.seed(seed)
    cov = poly_cov(p,0.7)
    x = random.multivariate_normal([0 for _ in range(p)],cov,size = n)
    x = random.normal(0,1,size = (n,p))
    x = np.concatenate(([[1] for i in range(n)],x),axis = 1)
    beta = [1,2,2,3,4,2]+[0 for i in range(p-5)]
    PI = 1/(1+np.exp(-x @ beta))
    y = random.binomial(1,np.reshape(PI,n),n)
    x = pd.DataFrame(x)
    x.columns = ['V'+str(i) for i in range((x.shape[1]))]
    x = x.iloc[:,1:]
    y=  pd.DataFrame(y)
    use_idx = [0,1,2,3,4]
    return x,y,PI,beta,use_idx

# %%
def get_performs(y,pred,pred_proba):
    cm1 = confusion_matrix(y,pred)
    total1=sum(sum(cm1))
    acc=(cm1[0,0]+cm1[1,1])/total1
    sens = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    spec = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    fpr, tpr, thresholds = roc_curve(y,pred_proba,pos_label= 1)
    a = auc(fpr,tpr)
    return acc,sens,spec,a

# %%
### Example ###
n1 =600;n2 = 100;p = 1000
X,Y,PI,_,use_idx = generate_data1(1,n1+n2,p)
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = n2)

# standardize
train_idx = x_train.index
test_idx = x_test.index
x_train = x_train.reset_index(drop = True)
x_train = (x_train-x_train.mean())/x_train.std()
x_test = x_test.reset_index(drop = True)
x_test = (x_test-x_train.mean())/x_train.std()
y_train = y_train.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)

# parameters setting
K = int(3*np.ceil(np.sqrt(n1/(np.log(p)))))
c1 = 1

# MPCGA+HDIC+MTrim
model = MPCGA(x_train,y_train,K,max_set = 3,max_split =3)
model.MPCGA_HDIC_Trim(c1,ic = 'BIC')
pred,pred_proba = model.MPCGA_pred(x_train,1,Mtrim=True)
acc,sens,spec,auc_ = get_performs(y_train,pred,pred_proba)

# MPCGA+HDIC
pred,pred_proba = model.MPCGA_pred(x_train,1)
acc,sens,spec,auc_ = get_performs(y_train,pred,pred_proba)
acc

# %%
### real data example
data_train = pd.read_csv('cancer_data8' +'/d_train'+'1.csv',index_col = 0)   
data_train = data_train.dropna(axis = 0,how = 'any')
x_train = data_train.iloc[:,6:].reset_index(drop = True)

# stadardize
n,p = x_train.shape
a = x_train.mean()
b = x_train.std()

x_train = (x_train-x_train.mean())/x_train.std()
x_train = x_train.dropna(axis = 1,how = 'all')
y_train = data_train[['lung_cancer']].reset_index(drop = True)

# paras setting
K = int(3*np.floor(np.sqrt(n/(np.log(p)))))
c1 = 0.7

#### algs start ####
model = MPCGA(x_train,y_train,K,max_set = 4,max_split =4)
model.MPCGA_HDIC_Trim(c1,ic = 'BIC')
pred,pred_proba = model.MPCGA_pred(x_train,1,Trim =False,Mtrim=True)
acc,sens,spec,a = get_performs(y_train,pred,pred_proba)
print(acc)
# %%
data_test = pd.read_csv('cancer_data8' +'/d_test'+'1.csv',index_col = 0)   
data_test = data_test.dropna(axis = 0,how = 'any')
x_test = data_test.iloc[:,6:].reset_index(drop = True)
x_test = (x_test-a)/b
x_test = x_test[x_train.columns]
y_test = data_test[['lung_cancer']].reset_index(drop = True)
pred,pred_proba = model.MPCGA_pred(x_test,Trim = False)
acc,sens,spec,a = get_performs(y_test,pred,pred_proba)
acc
# %%
data_test = pd.read_csv('cancer_data8' +'/d_test'+'2.csv',index_col = 0)   
data_test = data_test.dropna(axis = 0,how = 'any')
x_test = data_test.iloc[:,6:].reset_index(drop = True)
x_test = (x_test-a)/b
x_test = x_test[x_train.columns]
y_test = data_test[['lung_cancer']].reset_index(drop = True)
pred,pred_proba = model.MPCGA_pred(x_test,Trim = False)
acc,sens,spec,a = get_performs(y_test,pred,pred_proba)
acc
# %%
