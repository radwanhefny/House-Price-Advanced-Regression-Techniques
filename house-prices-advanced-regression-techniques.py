# ============================================================
# House Prices Regression | End-to-End ML Pipeline
# ============================================================

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import category_encoders as ce


# ============================================================
# Data Loading
# ============================================================

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')


# ============================================================
# Outlier Removal (EDA-driven)
# ============================================================

outlier_ids = [
    935, 1299, 250, 314, 336, 707, 379, 1183, 692, 186, 441,
    524, 739, 598, 955, 636, 49, 810, 1062, 1191, 496, 198
]

train_df = train_df[~train_df.Id.isin(outlier_ids)].reset_index(drop=True)


# ============================================================
# Missing Value Handling (Meaning-aware)
# ============================================================

no_cols = [
    'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu',
    'GarageCond', 'GarageType', 'GarageFinish', 'GarageQual',
    'BsmtExposure', 'BsmtQual', 'BsmtCond'
]

for col in no_cols:
    train_df[col] = train_df[col].fillna('No')
    test_df[col] = test_df[col].fillna('No')

zero_cols = ['LotFrontage', 'MasVnrArea']
for col in zero_cols:
    train_df[col] = train_df[col].fillna(0)
    test_df[col] = test_df[col].fillna(0)

unf_cols = ['BsmtFinType1', 'BsmtFinType2']
for col in unf_cols:
    train_df[col] = train_df[col].fillna('Unf')
    test_df[col] = test_df[col].fillna('Unf')

train_df['Electrical'] = train_df['Electrical'].fillna('SBrkr')
test_df['Electrical'] = test_df['Electrical'].fillna('SBrkr')


# ============================================================
# Target Transformation
# ============================================================

train_df['SalePrice'] = np.log1p(train_df['SalePrice'])


# ============================================================
# Ordinal Encoding
# ============================================================

all_mappings = {
    'Qual_map': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0, 'No': 0},
    
    'BsmtFinType_map': {'GLQ': 3, 'ALQ': 2, 'Unf': 2, 'BLQ': 1, 'Rec': 1, 'LwQ': 1, 'No': 0, 'NA': 0},
    'BsmtExposure_map': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'No_Bsmt': 0},
    'BsmtCond_map': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0, 'No': 0},
    
    'GarageFinish_map': {'Fin': 3, 'RFn': 2, 'Unf': 1, 'No': 0},
    'GarageQual_map': {'Gd': 3, 'Ex': 2, 'TA': 2, 'Fa': 1, 'Po': 0, 'No': 0},
    'GarageCond_map': {'Gd': 2, 'TA': 2, 'Ex': 1, 'Fa': 1, 'Po': 1, 'No': 0},
    'GarageType_map': {'BuiltIn': 3, '2Types': 2, 'Basment': 2, 'Attchd': 2, 'Detchd': 1, 'CarPort': 0, 'No': 0},
    
    'LandSlope_map': {'Gtl': 0, 'Mod': 1, 'Sev': 0},
    'LandContour_map': {'HLS': 3, 'Low': 2, 'Lvl': 1, 'Bnk': 0},
    'LotShape_map': {'Reg': 0, 'IR1': 1, 'IR2': 1, 'IR3': 1},
    'MSZoning_map': {'FV': 2, 'RL': 1, 'C (all)': 0, 'RM': 0, 'RH': 0},
    
    'Foundation_map': {'PConc': 2, 'Wood': 2, 'Stone': 2, 'CBlock': 1, 'BrkTil': 1, 'Slab': 0},
    'Electrical_map': {'SBrkr': 1, 'FuseF': 0, 'FuseA': 0, 'FuseP': 0, 'Mix': 0},
    'HeatingQC_map': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 2, 'No': 0},
    'CentralAir_map': {'Y': 1, 'N': 0},
    
    'Functional_map': {'Typ': 3, 'Min1': 1, 'Min2': 1, 'Mod': 1, 'Maj1': 1, 'Maj2': 0, 'Sev': 1, 'Sal': 0},
    'HouseStyle_map': {'2Story': 2, '2.5Fin': 2, 'SLvl': 2, '1Story': 1, '1.5Fin': 1, 'SFoyer': 1, '2.5Unf': 1, '1.5Unf': 0},
    'MasVnrType_map': {'Stone': 2, 'BrkFace': 1, 'BrkCmn': 0, 'No': 0},
    'ExterCond_map': {'Ex': 3, 'TA': 3, 'Gd': 2, 'Fa': 1, 'Po': 1, 'No': 0},
    'PavedDrive_map': {'Y': 2, 'P': 1, 'N': 0},
    'Alley_map': {'Pave': 2, 'No': 1, 'Grvl': 0},
    'Condition1_map': {'PosA': 2, 'PosN': 2, 'RRNe': 1, 'RRNn': 1, 'Norm': 1, 'RRAe': 0, 'RRAn': 0, 'Feedr': 0, 'Artery': 0}
}

def apply_ordinal_mappings(df, maps):
    qual_cols = ['ExterQual', 'BsmtQual', 'KitchenQual', 'FireplaceQu']
    for col in qual_cols:
        df[col] = df[col].map(maps['Qual_map']).fillna(0)
        
    for col in ['BsmtFinType1', 'BsmtFinType2']:
        df[col] = df[col].map(maps['BsmtFinType_map']).fillna(0)
        
    single_mappings = {
        'HeatingQC': 'HeatingQC_map', 'BsmtExposure': 'BsmtExposure_map',
        'PavedDrive': 'PavedDrive_map', 'GarageFinish': 'GarageFinish_map',
        'CentralAir': 'CentralAir_map', 'LandSlope': 'LandSlope_map',
        'LandContour': 'LandContour_map', 'LotShape': 'LotShape_map',
        'Functional': 'Functional_map', 'Alley': 'Alley_map',
        'Foundation': 'Foundation_map', 'HouseStyle': 'HouseStyle_map',
        'MasVnrType': 'MasVnrType_map', 'GarageType': 'GarageType_map',
        'MSZoning': 'MSZoning_map', 'Condition1': 'Condition1_map',
        'ExterCond': 'ExterCond_map', 'GarageQual': 'GarageQual_map',
        'Electrical': 'Electrical_map', 'BsmtCond': 'BsmtCond_map',
        'GarageCond': 'GarageCond_map'
    }
    
    for col, map_name in single_mappings.items():
        df[col] = df[col].map(maps[map_name]).fillna(0)
        
    return df

train_df = apply_ordinal_mappings(train_df, all_mappings)
test_df = apply_ordinal_mappings(test_df, all_mappings)

# ============================================================
# Ordinal Encoding
# ============================================================

mappings = {
    'Neighborhood': {k: 'Rare_N' for k in ['MeadowV', 'Blmngtn', 'BrDale', 'Veenker', 'NPkVill', 'Blueste']},
    'Exterior1st': {k: 'Rare_Ext' for k in ['AsbShng', 'BrkComm', 'Stone', 'AsphShn', 'ImStucc', 'CBlock']},
    'Exterior2nd': {k: 'Rare_Ext' for k in ['AsbShng', 'ImStucc', 'Brk Cmn', 'Stone', 'AsphShn', 'Other', 'CBlock']},
    'LotConfig': {'CulDSac': 'CulDSac_FR3', 'FR3': 'CulDSac_FR3'},
    'RoofStyle': {k: 'Other' for k in ['Gambrel', 'Flat', 'Mansard', 'Shed']}
}

for col, mapping in mappings.items():
    train_df[col] = train_df[col].replace(mapping)
    test_df[col] = test_df[col].replace(mapping)

def refine_sales(df):
    df['SaleCondition'] = df['SaleCondition'].apply(
        lambda x: 'Sale_Discount' if x in ['Abnorml', 'AdjLand']
        else 'Sale_High' if x == 'Partial'
        else 'Sale_Normal'
    )

    df['SaleType'] = df['SaleType'].apply(
        lambda x: 'Sale_High' if x in ['Con', 'New']
        else 'Sale_Normal' if x in ['WD', 'CWD']
        else 'Sale_Discount'
    )
    return df

train_df = refine_sales(train_df)
test_df = refine_sales(test_df)

# ============================================================
# Feature Engineering
# ============================================================

def create_features(df):

    df['houseage'] = df['YrSold'] - df['YearBuilt']
    df['houseremodelage'] = df['YrSold'] - df['YearRemodAdd']
    df['totalarea'] = df['GrLivArea'] + df['TotalBsmtSF']
    df['totalbaths'] = df['FullBath'] + df['BsmtFullBath'] + 0.5 * (df['BsmtHalfBath'] + df['HalfBath'])
    df['totalporchsf'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    
    return df

train_df = create_features(train_df)
test_df = create_features(test_df)

old_features = ['YrSold', 'YearBuilt', 'YearRemodAdd', 'GrLivArea', 'TotalBsmtSF',
                'FullBath', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'OpenPorchSF',
                '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF']

train_df.drop(columns=old_features, inplace=True, errors='ignore')
test_df.drop(columns=old_features, inplace=True, errors='ignore')

# ============================================================
# Feature Selection & Dimensionality Reduction
# ============================================================

cols_to_drop_missing = ['PoolQC', 'MiscFeature', 'Fence'] 
cols_to_drop_dominant = ['Street', 'Utilities', 'Condition2', 'Heating', 'RoofMatl']
cols_to_drop_corr = ['GarageArea', 'GarageYrBlt', 'GarageCond', 'Fireplaces', 'Id']

final_drop_list = cols_to_drop_missing + cols_to_drop_dominant + cols_to_drop_corr 

train_df.drop(columns=final_drop_list, inplace=True, errors='ignore')
test_df.drop(columns=final_drop_list, inplace=True, errors='ignore')

# ============================================================
# Train / Validation Split
# ============================================================

X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=25
)


# ============================================================
# Preprocessing Pipeline
# ============================================================

targetE_cols = ['Neighborhood', 'Exterior1st', 'Exterior2nd']
ohe_cols = ['LotConfig','RoofStyle', 'SaleCondition', 'BldgType', 'SaleType']
num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
num_cols = num_cols.drop('SalePrice')

num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

targetE_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('tergetE', ce.TargetEncoder(smoothing=10))
])

ohe_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

col_trans = ColumnTransformer(transformers=[
    ('num_p', num_pipeline, num_cols),
    ('targetE_p', targetE_pipeline, targetE_cols),
    ('ohe_p', ohe_pipeline, ohe_cols)],
    remainder='drop', n_jobs=-1)

pipeline = Pipeline(steps=[
    ('preprocessing', col_trans)])

X_train_preprocessed = pipeline.fit_transform(X_train, y_train)
X_test_preprocessed = pipeline.transform(X_test)


# ============================================================
# Models & Ensemble
# ============================================================

final_ridge = Ridge(alpha=11.015056790269638, solver='cholesky', random_state=42)

final_xgb = XGBRegressor(n_estimators=2344,
                         max_depth=3,
                         learning_rate=0.010995780706257275,
                         subsample=0.5032771700290668,
                         colsample_bytree=0.6532582074342806,
                         reg_alpha=0.39975402381338976,
                         reg_lambda=2.062558657922061,
                         min_child_weight=3,
                         random_state=42, 
                         verbosity=0)

final_gbr = GradientBoostingRegressor(n_estimators=2899, 
                                      max_depth=4, 
                                      learning_rate=0.0037873524294740543, 
                                      subsample=0.3401227521784775, 
                                      max_features=0.25868425566686204,
                                      min_samples_leaf=5,
                                      min_samples_split=17,
                                      random_state=42)

final_lgbm = lgb.LGBMRegressor(n_estimators=2059, 
                               learning_rate=0.005627035068472362,
                               num_leaves=26,
                               min_child_samples=12,
                               reg_alpha=0.11889380370039195,
                               reg_lambda=0.029531782415710237,
                               colsample_bytree=0.3672036876746482,
                               subsample=0.6004305308267477,
                               subsample_freq=2,
                               random_state=42, 
                               verbosity=-1)

final_cat = CatBoostRegressor(iterations=2003,
                          depth=3,
                          learning_rate=0.02093119419796248,
                          random_strength=9.49832943872932,
                          bagging_temperature=0.15705109755836116,
                          border_count=98,
                          l2_leaf_reg=3.9259923205055367,
                          random_state=42, 
                          verbose=0)

estimators = [
    ('ridge', final_ridge),
    ('gbr', final_gbr),
    ('cat', final_cat),
    ('xgb', final_xgb),
    ('lgb', final_lgbm)
]

vr = VotingRegressor(estimators=estimators, weights=[0.40, 0.18, 0.22, 0.10, 0.10])
vr.fit(X_train_preprocessed, y_train)
y_pred_vr = vr.predict(X_test_preprocessed)

print("Validation RMSE:",
      root_mean_squared_error(y_test, y_pred_vr))


# ============================================================
# Final Training & Prediction
# ============================================================

X_train_final = pipeline.fit_transform(X, y)
vr_final = VotingRegressor(estimators=estimators, weights=[0.40, 0.18, 0.22, 0.10, 0.10])
vr_final.fit(X_train_final, y)

df_test_preprocess = pipeline.transform(test_df)
final_blend_comp = vr_final.predict(df_test_preprocess)
final_blend_comp = np.clip(final_blend_comp, 10.5, 14.5)
y_final_prices = np.expm1(final_blend_comp)

test_df_original = pd.read_csv('data/test.csv')
df_out = pd.DataFrame()

df_out['Id'] = test_df_original['Id']
df_out['SalePrice'] = y_final_prices

df_out.to_csv('submission.csv', index=False)
print("submission.csv generated successfully")
