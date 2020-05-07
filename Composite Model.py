# %% [markdown]
# # Composite Model
"""
A composite model will be constructed where the features associated with each fight will be 
predicted by models trained on each fighters historical fight data. For example a model will
be trained to predict the the number of strikes landed by the fighter in the red corner.

A model will then be trained to predict the outcome of fights based on the predictions of the
previous models.
"""
# ## Data Preperation
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from sklearn.externals import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

# %% [markdown]
# Keep only raw features not the constructed ones and let's define Historic Features
# and features associated with the fight, Fight Features. We also need to scale the historic
# as this will be used for training.
# %%
data = pd.read_csv("FinalData.csv")
AllFeatures = data.columns.to_list()[:data.columns.to_list().index("R_TD_ELR")]
data = data[AllFeatures].dropna()

FightFeatrues = AllFeatures[AllFeatures.index('R_SIG_STR._landed'):AllFeatures.index('R_total_no._fights')]
HistoricFeatures = AllFeatures[AllFeatures.index(
    'R_total_no._fights'):]

FightData = data[FightFeatrues]
FightData = (FightData[FightData.columns[FightData.columns.str.contains("R_|B_")]]
             .select_dtypes(exclude="object"))

HistoricData = StandardScaler().fit_transform(
    data[HistoricFeatures].select_dtypes(exclude="object"))
    
#%% [markdown]
"""
Now let's create the testing and training data and then transform it. 
"""

#%%

TrainFightData, TestFightData, TrainHistoricData, TestHistoricData = train_test_split(
    FightData, HistoricData, test_size=0.2, random_state=3943
)

HistoricTransformer = StandardScaler()
TrainHistoricData = HistoricTransformer.fit_transform(TrainHistoricData)
TestHistoricData = HistoricTransformer.transform(TestHistoricData)



# %% [markdown]
# ## Fight Feature Model Training
"""
Now let's train the fight feature models. We will train a variety of models for each feature. 
begin with some support vector machines. 

"""

# %%
GridParams = {
    "cv": 5,
    "verbose": 10,
    "n_jobs": -1,
    "return_train_score": False
}


ModelsDataFrame = pd.DataFrame(columns=FightData.columns.to_list(),
    index=["SVM","PolySVM"])


for index,feature in enumerate(FightData.columns):
    print("\n - - - - - - - - - - Training",
          feature, "Models - - - - - - - - - - ")
    print(" - - - - - Feature",index+1,"out of",len(FightData.columns)," - - - - - ")

    FeatureModels = ({
    "SVM": GridSearchCV(SVR(kernel="rbf"), **GridParams,
                        param_grid={
        "C": np.logspace(0, 3, num=25),
    }),

    "PolySVM": GridSearchCV(SVR(kernel="poly"), **GridParams,
                            param_grid={
        "degree": [2, 4, 6],
        "C": np.logspace(0, 3, num=15)
    })
    })

    for ModelName, model in FeatureModels.items():
        print("                * Training", ModelName, "*")
        model.fit(TrainHistoricData,
                  TrainFightData[feature].values)
        ModelsDataFrame[feature][ModelName] = model


joblib.dump(ModelsDataFrame, "models/SVM_Model_DF.pkl")

# %%

SVM_Models_DF = joblib.load("models/SVM_Model_DF.pkl")

SVM_Accuracy = SVM_Models_DF.applymap(lambda model:model.best_score_)
SVM_Params = SVM_Models_DF.applymap(lambda model: model.best_params_)
display(SVM_Accuracy)
display(SVM_Params)

#%% [markdown]
"""
Now lets train some gradient boosted models. We will use the K-Fold cross validation with early stopping.
"""
#%%

params = {
    "n_iter_no_change": 5
}

ParamGrid ={
    "max_depth":[1,2,3,6,12],

}