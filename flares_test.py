import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler  # 归一化
import kmapper as km
import umap.umap_ as umap
import sklearn
import sklearn.manifold as manifold
# import matplotlib.pyplot as plt
# from kmapper.jupyter import display
from sklearn.decomposition import PCA
from sklearn import cluster
import networkx as nx

# 读入数据
player = pd.read_csv('datacleaning/modified_data.csv', header=0, encoding='unicode_escape', delimiter=';')
player.head(6)

# z score 归一化
scaler = StandardScaler()
for col in player.columns:
    if col in ('Rk', 'Player', 'Nation', 'Pos', 'Comp', 'Age', 'Born', 'Squad'): continue
    player[[col]] = scaler.fit_transform(player[[col]])

# squad = player['Squad']
# season = player['Season']
# pos = player['Pos']
# values = player.drop(['Squad', 'Season'], axis=1).values
# comp_dim = kargs['pca_dim']
# pca = PCA(n_components=comp_dim)  # 15
# values = pca.fit_transform(values)
# # print(sum(pca.explained_variance_ratio_))
# player = pd.DataFrame(values)
# player['Squad'] = squad
# player["Season"] = season
# player['Pos'] = pos

# initialize mapper
mapper = km.KeplerMapper(verbose=0)
# project data
X_features_numeric = player.drop(['Rk', 'Player', 'Nation', 'Pos', 'Comp', 'Age', 'Born', 'Squad'],
                                           axis=1).values  # data array
projected_X = mapper.fit_transform(X_features_numeric,
                                   projection=[manifold.Isomap(n_components=10, n_jobs=-1),
                                               umap.UMAP(n_components=2, random_state=1)], scaler=[None])
graph2 = mapper.map(projected_X, cover=km.Cover(n_cubes=15),
                    clusterer=sklearn.cluster.KMeans(n_clusters=5, random_state=1618033))

'''
# 根据标签数据y创建对应的color_values
def create_color_values(y):
    unique_labels = np.unique(y)
    color_values = [str(i + 1) for i in range(unique_labels.shape[0])]
    label_to_color = dict(zip(unique_labels, color_values))
    return np.array([label_to_color[label] for label in y])


def node_color_func(node):
    labels, counts = np.unique(node, return_counts=True)
    max_count_index = np.argmax(counts)
    ans = create_color_values(labels)
    return ans[max_count_index]
'''

pos = np.array(player['Pos'])
graph = km.adapter.to_nx(graph2)
clusters = graph.nodes.data()

# 将mapper的点和原数据点对应
node_dict = {}
for clusterdata in clusters:
    cube_cluster = clusterdata[0]
    members = clusterdata[1]["membership"]
    for member in members:
        node_dict[member] = cube_cluster
print(node_dict)

# 探测flare
# import mappertools.features.flares as flr
import flares as flr
cen = nx.centrality.closeness_centrality(graph)
flares = flr.flare_detect(graph,cen)
long_flares = flr.threshold_flares(flares,0)
print(flares)
print(long_flares)

# 定义一个函数，用于计算Flare对象的生命周期，如果death为None，返回-1
def get_lifetime(flare):
    if flare.death is None:
        return -1
    else:
        return flare.death - flare.birth

# 对每个Flare对象，计算它的生命周期lifetime
for flare in flares:
    flare.lifetime = get_lifetime(flare)

# 定义一个空的字典存储每个点的生命周期
point_lifetime = {}
# 如果一个member属于多个Flare对象，取最大的lifetime值
for flare in flares:
    for member in flare.nodes:
        if member not in point_lifetime or point_lifetime[member] < flare.lifetime:
            point_lifetime[member] = flare.lifetime

# 查看每个点对应的生命周期值
print(point_lifetime)

# 将原数据点和所在flare的生命周期对应
lifetime_node_dict = {}
for index, cluster_belong in node_dict.items():
    lifetime_node_dict[index] = point_lifetime[cluster_belong]
print(lifetime_node_dict)

lifetime_series = pd.Series(lifetime_node_dict,name="flare_lifetime")
new_data = player.join(lifetime_series, how='left', rsuffix='_lifetime').fillna(0)
new_data[["flare_lifetime"]]

#lifetime不等于-1的
new_data_drop = new_data[new_data['flare_lifetime'] !=-1]
print(len(new_data_drop))

#lifetime等于-1的
new_data_nega = new_data[new_data['flare_lifetime'] ==-1]

# 找出数值型属性列，主成分分析出15个指标(除flare_lifetime外)
from scipy.stats import spearmanr, kendalltau
num_cols = new_data_drop.select_dtypes(include=np.number).columns
num_cols = num_cols.drop(['Rk','Age','Born', "flare_lifetime"])
pca = PCA(n_components=15)
pca.fit(new_data_drop[num_cols])
factors = pca.transform(new_data_drop[num_cols])
# loadings = pca.components_
# print(loadings)
# 将15个主成分另做一个dataframe，一一与flare_lifetime进行相关分析
factors_df = pd.DataFrame(factors, columns=["factor"+str(i) for i in range(1,16)])
factors_df["flare_lifetime"] = new_data_drop["flare_lifetime"]
count = 0
for column in factors_df.columns:
    spearman_corr, spearman_p = spearmanr(new_data_drop["flare_lifetime"], factors_df[[column]])
    if spearman_p <= 0.01 and spearman_corr > 0:
        count += 1
        print(f"correlation between\n {column} and flare_lifetime:")
        print(f"spearman_corr:{spearman_corr},spearman_p:{spearman_p} \n")
print(count)

#与数值变量进行相关分析
count = 0
for column in new_data_drop:
    if column in ('Rk', 'Player', 'Nation', 'Pos', 'Comp', 'Age', 'Born', 'Squad'):
        continue
    spearman_corr, spearman_p = spearmanr(new_data_drop["flare_lifetime"], new_data_drop[[column]])
    if spearman_p <= 0.01 and spearman_corr <0:
        count +=1
        print(f"correlation between\n {column} and flare_lifetime:")
        print(f"spearman_corr:{spearman_corr},spearman_p:{spearman_p} \n")
print(count)
# 共85个的p检验数小于0.01，其中19个呈现正相关，66个呈现负相关


# 导入scipy库
from scipy.stats import f_oneway

# 对pos的每个水平，提取flare_lifetime的值，得到一个列表
y_list = [new_data_drop['flare_lifetime'][new_data_drop['Pos'] == level] for level in new_data_drop['Pos'].unique()]

# 进行单因素方差分析，得到F值和p值
F, p = f_oneway(*y_list)

# 打印结果
print('F-value:', F)
print('p-value:', p)
# p值相当小，有显著差异
#即球员的位置和flare_lifetime显著相关

# kruskal检验
from scipy.stats import kruskal
y_list = [new_data_drop['flare_lifetime'][new_data_drop['Pos'] == level] for level in new_data_drop['Pos'].unique()]
# 进行Kruskal-Wallis检验，得到H值和p值
H, p = kruskal(*y_list)
# p值相当小，有显著差异

# 打印结果
print('H-value:', H)
print('p-value:', p)