import numpy as np
import pandas as pd
import re as re

from base import Feature, get_arguments, generate_features
from target_encoder import KFoldTargetEncoderTrain, TargetEncoderTest

from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
Feature.dir = 'features'


def fixing_skewness(df):
    """
    This function takes in a dataframe and return fixed skewed dataframe
    """
    # Getting all the data that are not of "object" type.
    numeric_feats = df.dtypes[df.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skewed_feats[abs(skewed_feats) > 0.5]
    skewed_features = high_skew.index
    for feat in skewed_features:
        df[feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))
    return df

def fill_features_na(df):
    return df.fillna(df.median())



class Numerical(Feature):
    def create_features(self):
        df = features[numeric_features].copy()
        df = fill_features_na(df)
        #df = df.drop(overfit_reducer(df), axis=1)
        df = fixing_skewness(df)
        self.train = df[:train.shape[0]]
        self.test = df[train.shape[0]:]


class Objects(Feature):
    def create_features(self):
        df = features[categorical_features].copy()
        df['gender'] = df['gender'].replace(['Female','Male'], [0, 1])
        df['ever_married'] = df['ever_married'].replace(['No','Yes'], [0, 1])
        df['Residence_type'] = df['Residence_type'].replace(['Rural','Urban'], [0, 1])
        for feature in categorical_features:
            df[feature] = df[feature].fillna("Missing")
        df = pd.get_dummies(df)
        self.train = df[:train.shape[0]]
        self.test = df[train.shape[0]:]

class Pca(Feature):    # 効果あり 1 だけでも良い
    def create_features(self):
        df = fill_features_na(features[numeric_features].copy()) # 欠損値を穴埋め
        scaler = StandardScaler()
        # scaler = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
        numeric_feats = df.dtypes[df.dtypes != "object"].index
        scaled_df = pd.DataFrame(scaler.fit_transform(df[numeric_feats]), columns=numeric_feats)
        n_pca = 2
        pca_cols = ["pca"+str(i) for i in range(n_pca)]
        pca = PCA(n_components=n_pca)
        pca_df = pd.DataFrame(pca.fit_transform(scaled_df), columns=pca_cols)
        pca_df[pca_cols] = scaler.fit_transform(pca_df[pca_cols])
        self.train = pca_df[:train.shape[0]].reset_index(drop=True)
        self.test = pca_df[train.shape[0]:].reset_index(drop=True)
        '''
        n_tsne = 2
        tsne_cols = ["tsne"+str(i) for i in range(n_tsne)]
        embeded = pd.DataFrame(bhtsne.tsne(all_df[numeric_features].astype(np.float64), dimensions=n_tsne, rand_seed=10), columns=tsne_cols)
        features = pd.concat([scaled_df, pca_df, embeded], axis=1)
        '''

class Tsne(Feature):    # 効果あり 1 だけでも良い
    def create_features(self):
        df = fill_features_na(features[numeric_features].copy()) # 欠損値を穴埋め
        scaler = StandardScaler()
        # scaler = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
        numeric_feats = df.dtypes[df.dtypes != "object"].index
        scaled_df = pd.DataFrame(scaler.fit_transform(df[numeric_feats]), columns=numeric_feats)
        n_pca = 2
        pca_cols = ["pca"+str(i) for i in range(n_pca)]
        pca = PCA(n_components=n_pca)
        pca_df = pd.DataFrame(pca.fit_transform(scaled_df), columns=pca_cols)
        pca_df[pca_cols] = scaler.fit_transform(pca_df[pca_cols])
        self.train = pca_df[:train.shape[0]].reset_index(drop=True)
        self.test = pca_df[train.shape[0]:].reset_index(drop=True)
        '''
        n_tsne = 2
        tsne_cols = ["tsne"+str(i) for i in range(n_tsne)]
        embeded = pd.DataFrame(bhtsne.tsne(all_df[numeric_features].astype(np.float64), dimensions=n_tsne, rand_seed=10), columns=tsne_cols)
        features = pd.concat([scaled_df, pca_df, embeded], axis=1)
        '''

class Polynomial2d(Feature): # うまくいった。もう少し絞るのが良さそう
    def create_features(self):
        df = fill_features_na(features[numeric_features].copy()) # 欠損値を穴埋め
        numeric_feats = df.dtypes[df.dtypes != "object"].index
        df = df[numeric_feats]
        poly = PolynomialFeatures(2)
        original_fea_num = df.shape[1]
        poly_np = poly.fit_transform(df)[:, original_fea_num+1:]
        poly_features = poly.get_feature_names(df.columns)[original_fea_num+1:]
        poly_df = pd.DataFrame(poly_np, columns=poly_features)
        # fixed skew
        poly_df = fixing_skewness(poly_df)
        self.train = poly_df[: train.shape[0]].reset_index(drop=True)
        self.test = poly_df[train.shape[0]:].reset_index(drop=True)

class Polynomial3d(Feature):
    def create_features(self):
        df = fill_features_na(features[numeric_features].copy()) # 欠損値を穴埋め
        numeric_feats = df.dtypes[df.dtypes != "object"].index
        df = df[numeric_feats]
        poly = PolynomialFeatures(3)
        original_fea_num = df.shape[1]
        poly_np = poly.fit_transform(df)[:, original_fea_num+1:]
        poly_features = poly.get_feature_names(df.columns)[original_fea_num+1:]
        poly_df = pd.DataFrame(poly_np, columns=poly_features)
        # fixed skew
        poly_df = fixing_skewness(poly_df)
        self.train = poly_df[: train.shape[0]].reset_index(drop=True)
        self.test = poly_df[train.shape[0]:].reset_index(drop=True)


if __name__ == '__main__':
    args = get_arguments()
    train_with_target = pd.read_feather('./data/interim/train.feather')
    train = train_with_target.drop(["id","stroke"],axis=1) # id,target を落とす
    test = pd.read_feather('./data/interim/test.feather')
    test = test.drop(["id"],axis=1) # id を落とす
    features = pd.concat([train, test])
    numeric_features = [feature for feature in features.columns if features[feature].dtype !=
                        'O' ]
    categorical_features = [feature for feature in features.columns if features[feature].dtype == 'O']
    print(features.head())
    print(features.info())

    generate_features(globals(), args.force)
