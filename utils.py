# utils.py
import numpy as np
import pandas as pd
def read_GithubData_engineered():
    # return a dataframe
    path = r'GithubData\engineered\outfield-goalkeeper-combined\fbref__outfield_player_goalkeeper_stats_combined_latest.csv'
    selected_cols = ['Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age', 'Born', 'MP', 'Starts', 
                     'Min', '90s', 'Gls', 'Ast', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR', 'Gls.1', 
                     'Ast.1', 'G+A', 'G-PK.1', 'G+A-PK', 'xG', 'npxG', 'xA', 'npxG+xA', 'xG.1', 'xA.1', 
                     'xG+xA', 'npxG.1', 'npxG+xA.1', 'Sh', 'SoT', 'SoT%', 'Sh/90', 'SoT/90', 
                     'G/Sh', 'G/SoT', 'Dist', 'FK', 'npxG/Sh', 'G-xG', 'np:G-xG', 'Cmp', 'Att', 'Cmp%', 
                     'TotDist', 'PrgDist', 'Cmp.1', 'Att.1', 'Cmp%.1', 'Cmp.2', 'Att.2', 'Cmp%.2', 'Cmp.3', 
                     'Att.3', 'Cmp%.3', 'A-xA', 'KP', '1/3', 'PPA', 'CrsPA', 'Prog', 'Live', 'Dead', 'TB', 
                     'Press', 'Sw', 'Crs', 'CK', 'In', 'Out', 'Str', 'Ground', 'Low', 'High', 'Left', 'Right', 
                     'Head', 'TI', 'Other', 'Off', 'Out.1', 'Int', 'Blocks', 'SCA', 'SCA90', 'PassLive', 'PassDead', 
                     'Drib', 'Fld', 'Def', 'GCA', 'GCA90', 'PassLive.1', 'PassDead.1', 'Drib.1', 'Sh.1', 'Fld.1', 
                     'Def.1', 'Tkl', 'TklW', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Tkl.1', 'Tkl%', 'Past', 'Succ', 
                     '%', 'Def 3rd.1', 'Mid 3rd.1', 'Att 3rd.1', 'ShSv', 'Pass', 'Tkl+Int', 'Clr', 'Err', 'Touches',
                       'Def Pen', 'Att Pen', 'Succ%', '#Pl', 'Megs', 'Carries', 'CPA', 'Mis', 'Dis', 'Targ',
                         'Rec', 'Rec%', 'Prog.1', 'Mn/MP', 'Min%', 'Compl', 'Subs',  # 'Mn/Sub', 'Mn/Start', 
                           'unSub', 'PPM', 'onG', 'onGA', '+/-', '+/-90', 'On-Off', 'onxG', 'onxGA', 'xG+/-', 
                           'xG+/-90', 'On-Off.1', '2CrdY', 'Fls', 'PKwon', 'PKcon', 'OG', 'Recov', 'Won', 'Lost',
                             'Won%', 'League Name', 'League ID', 'Season', 'Team Name', 'Team Country', 'Player Lower', 
                             'First Name Lower', 'Last Name Lower', 'First Initial Lower', 'Team Country Lower', 
                             'Nationality Code', 'Nationality Cleaned', 'Position Grouped', 'Outfielder Goalkeeper']
                        # 清除这些不需要的，残缺的，只有部分守门员有的数据
                       #        'GA', 'GA90', 'SoTA', 'Saves', 'Save%', 'W', 'D', 'L', 'CS', 'CS%', 'PKA', 'PKsv', 'PKm', 'Save%.1', 'PSxG', 'PSxG/SoT', 'PSxG+/-', '/90', 'Thr', 'Launch%', 'AvgLen', 'Launch%.1', 'AvgLen.1', 'Opp', 'Stp', 'Stp%', '#OPA', '#OPA/90', 'AvgDist']
    df = pd.read_csv(path, header=0, encoding='utf-8')
    # 生成season列表示赛季
    df['Season'] = df['Age'] + df['Born']
    df = df[selected_cols]
    df = df[df['MP'] > 4]
    return df


def read_GithubData_raw(year):
    # return 2 df
    assert year in range(2017, 2022)
    GoalKeeperPath = f'GithubData/raw/goalkeeper/Big-5-European-Leagues/{year}-{year+1}/fbref_goalkeeper_stats_Big-5-European-Leagues_{year}-{year+1}_latest.csv'
    outfieldPath = f'GithubData/raw/outfield/Big-5-European-Leagues/{year}-{year+1}/fbref_outfield_player_stats_Big-5-European-Leagues_{year}-{year+1}_latest.csv'
    df1 = pd.read_csv(GoalKeeperPath, header=0, encoding='utf-8')
    df2 = pd.read_csv(outfieldPath, header=0, encoding='utf-8')
    return df1, df2

def read_GithubData_mycombine_outfielder():
    df = pd.read_excel('DLdata/outfielder_combined_from_raw.xlsx')
    selected_cols = ['Player', 'Pos', 'Squad', 'Comp', 'Age', 'Born', 'MP', 'Starts', 'Min', 
                     '90s', 'Gls', 'Ast', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR', 'Gls.1', 'Ast.1', 
                     'G+A', 'G-PK.1', 'G+A-PK', 'xG', 'npxG', 'xA', 'npxG+xA', 'xG.1', 'xA.1', 
                     'xG+xA', 'npxG.1', 'npxG+xA.1', 'Sh', 'SoT', 'SoT%', 'Sh/90', 'SoT/90', 
                     'G/Sh', 'G/SoT', 'Dist', 'FK', 'npxG/Sh', 'G-xG', 'np:G-xG', 'Cmp', 'Att', 'Cmp%',
                    'TotDist', 'PrgDist', 'Cmp.1', 'Att.1', 'Cmp%.1', 'Cmp.2', 'Att.2', 'Cmp%.2', 
                    'Cmp.3', 'Att.3', 'Cmp%.3', 'A-xA', 'KP', '1/3', 'PPA', 'CrsPA', 'Prog', 'Live',
                    'Dead', 'TB', 'Press', 'Sw', 'Crs', 'CK', 'In', 'Out', 'Str', 'Ground',
                    'Low', 'High', 'Left', 'Right', 'Head', 'TI', 'Other', 'Off', 'Out.1',
                     'Int', 'Blocks', 'SCA', 'SCA90', 'PassLive', 'PassDead', 'Drib', 'Fld', 
                     'Def', 'GCA', 'GCA90', 'PassLive.1', 'PassDead.1', 'Drib.1', 'Sh.1', 'Fld.1',
                       'Def.1', 'Tkl', 'TklW', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Tkl.1', 'Tkl%',
                     'Past', 'Succ', '%', 'Def 3rd.1', 'Mid 3rd.1', 'Att 3rd.1', 'ShSv', 'Pass',
                     'Tkl+Int', 'Clr', 'Err', 'Touches', 'Def Pen', 'Att Pen', 'Succ%',
                     '#Pl', 'Megs', 'Carries', 'CPA', 'Mis', 'Dis', 'Targ', 'Rec', 'Rec%',
                     'Prog.1', 'Mn/MP', 'Min%', 'Compl', 'Subs', 'unSub',
                 'PPM', 'onG', 'onGA', '+/-', '+/-90', 'On-Off', 'onxG', 'onxGA', 'xG+/-', 'xG+/-90',
             'On-Off.1', '2CrdY', 'Fls', 'PKwon', 'PKcon', 'OG', 'Recov', 'Won', 'Lost', 'Season']
    df = df[selected_cols]
    df = df[df['MP'] > 4]

    # 随机森林填补缺失值
    from sklearn.ensemble import RandomForestRegressor
    #from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.impute import SimpleImputer
    #na = df.isna().sum()
    na_cols = [col for col in df.columns if df[col].isnull().any()]
    print(na_cols)

    
    df_data = df.drop(['Player', 'Pos', 'Squad', 'Comp', 'Age', 'Born', 'Season'], axis=1)  # 保证col是能输入模型的
    training = df_data.dropna(axis=0)  # 删除行
    to = len(na_cols)
    fini = 1
    for col in na_cols:
        #print(len(na_cols))
        Xtrain = training.drop(col, axis=1)
        Ytrain = training[col]
        #X_train ,X_test, y_train, y_test = train_test_split(Xtrain, Ytrain, test_size=0.3, random_state=0)
        # 实例化随机森林回归函数类
        forest_clf = RandomForestRegressor(random_state=42)
        # 建立需要搜索的参数的范围
        param_grid =[{'n_estimators':[10,30,50,100],
                    'max_depth':[5,10,20]}]
        # 初始化网格搜索的方法
        grid_search = GridSearchCV(forest_clf, param_grid, cv=3, n_jobs=7, scoring='neg_mean_squared_error')  # cv5意味着使用5折交叉验证
        #用网格搜索方法进行拟合数据
        grid_search.fit(Xtrain, Ytrain)
        # 输出最优的参数组合
        print(f'{fini}/{to} col: {col}\n最佳参数:', end=' ')
        print(grid_search.best_params_)
        print(f'score: {grid_search.best_score_}')
        #print(f'cv result: {grid_search.cv_results_}')
        best_model = grid_search.best_estimator_

        # 下面进行填补

        tofill = df[df[[col]].isnull().any(axis=1)]
        tofill = tofill.drop([col] + ['Player', 'Pos', 'Squad', 'Comp', 'Age', 'Born', 'Season'], axis=1)
        tofill = tofill.fillna(value=tofill.mean()) #先用均值填补其他的缺失值，避免对随机森林模型造成影响 
        tofill = tofill.fillna(value=0)             # 可能存在所有列的都缺失的情况，所以用0再填一次

        Ypredict = best_model.predict(tofill)
        #new_col = df[col]   # 在最原始的数据上做修改
        #new_col[new_col.isna()] = Ypredict
        df.loc[df.loc[: ,col].isnull(), col] = Ypredict

        fini += 1

    df.to_csv('DLdata/outfielder_combined_from_raw_fillna.csv', encoding='utf-8')
    return df


def read_Kaggle(year):
    assert year in [2021, 2022]
    path = f'KaggleData/view.xlsx'
    df = pd.read_excel(path, sheet_name=f'{year}-{year+1}', header=0, index_col='Rk')
    return df


def robust_name_match(name1, name2, LENRANGE = 0.8, MATCHRATE = 0.8):
    # 因为编码问题存在部分名字读取部分错误的情况，所以得写这个函数
    if len(name1) > len(name2):
        name1, name2 = name2, name1 # 确保name1短于name2
    if len(name1) / len(name2) < LENRANGE:  # 长度差异悬殊
        return False
    
    l = len(name1)
    d = len(name2) - len(name1)
    
    for times in range(d + 1):
        curmatch = 0
        for i in range(l):
            if name1[i] == name2[i + times]:
                curmatch += 1

        if (curmatch / l) >= MATCHRATE:
            return True  # 若本次迭代已经满足要求
    return False
    

def randomSplit(data: np.array, part=(0.5, 0.7, 1.0)):
    lenth = data.shape[0]
    l = list(range(lenth))
    np.random.shuffle(l)
    train = data[l[:int(part[0]*lenth)], :]
    valid = data[l[int(part[0]*lenth):int(part[1]*lenth)], :]
    test = data[l[int(part[1]*lenth):int(part[2]*lenth)], :]
    return train, valid, test


class Accumulator():
    def __init__(self, c):
        self.data = []
        for i in range(c):
            self.data.append([])
        self.c = c
        self.backup = []
    def add(self, *args):
        assert len(args) == self.c
        for i in range(self.c):
            self.data[i].append(args[i])
            
                
    def clear(self):  
        '''把上一次的累积平均求出来，然后存入backup'''
        if len(self.data[0]) > 0:
            self.backup.append([sum(l)/len(l) for l in self.data])
            self.data = []
            for i in range(self.c):
                self.data.append([])
        
    def __getitem__(self, idx):
        '''获得当前累积的几个指标的平均值'''
        if len(self.data[idx]) > 0:
            return sum(self.data[idx]) / len(self.data[idx])
        return 0

    def read_backup(self, id):
        '''获取第几列上的backup数据'''
        return [l[id] for l in self.backup]
    

from torch_geometric.data import Data
import torch
from torch import nn
import torch.nn.functional as F
import random
# 我自己写一个position embedding 的模块吧
class PosiEmbedding(nn.Module):
    def __init__(self, node_size, embed_size):
        super(PosiEmbedding, self).__init__()
        self.node_size = node_size
        self.embed_size = embed_size
        self.in_embed = nn.Embedding(self.node_size, self.embed_size)  # 中心词权重矩阵
        self.out_embed = nn.Embedding(self.node_size, self.embed_size)  # 周围词权重矩阵
        # 节点向量层参数初始化
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)
    def forward_input(self, input_labels):
        '''
            input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, (window_size * 2 * K)]
            return: embed, [batch_size]
            '''
        input_embedding = self.in_embed(input_labels) # [batch_size, words_count, embed_size]
        return input_embedding
    def forward_target(self, out_embedding):
        out_embedding = self.out_embed(out_embedding)# [batch_size, (window * 2), embed_size]
        return out_embedding
    def input_embedding(self):
        return self.in_embed.weight.detach().numpy()
    def forward(self, input_labels):
        input_embedding = self.in_embed(input_labels) # [batch_size, embed_size]
        out = torch.matmul(input_embedding, torch.transpose(self.out_embed.weight.detach(), 0, 1))
        s = nn.Softmax(dim=1)  # 在第一个维度求
        return s(out)
class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_embedding, pos_embedding, neg_embedding):
        '''
            input_labels: center words, [batch_size, 1, embed_size]
            pos_labels: positive words, [batch_size, (window * 2), embed_size]
            neg_labels：negative words, [batch_size, (window * 2 * K), embed_size]
            
            return: loss, [batch_size]
        '''
        # squeeze是挤压的意思，所以squeeze方法是删除一个维度，反之，unsqueeze方法是增加一个维度
        # bmm方法是两个三维张量相乘，两个tensor的维度是，（b * m * n）, (b * n * k) 得到（b * m * k），相当于用矩阵乘法的形式代替了网络中神经元数量的变化
        # 矩阵的相乘相当于向量的点积，代表两个向量之间的相似度
        input_embedding = torch.transpose(input_embedding, 1, 2) # 将第一维和第二维换一下，变成 [batch_size, embed_size, 1]
        #print(input_embedding.shape)
        #print(pos_embedding.shape)
        pos_dot = torch.matmul(pos_embedding, input_embedding)  # [batch_size, (window * 2), 1]
        #print(pos_dot.shape)
        #input()
        pos_dot = pos_dot.squeeze(2)  # [batch_size, (window * 2)]
        neg_dot = torch.matmul(neg_embedding, -input_embedding)  # [batch_size, (window * 2 * K), 1]，这里之所以用减法是因为下面loss = log_pos + log_neg，log_neg越小越好
        neg_dot = neg_dot.squeeze(2)  # batch_size, (window * 2 * K)]
        log_pos = F.logsigmoid(pos_dot).sum(1)  # .sum()结果只为一个数，.sum(1)结果是一维的张量，在序号1的维度求和
        log_neg = F.logsigmoid(neg_dot).sum(1)  # 这两个loss都是[batch_size]的张量
        loss = log_pos + log_neg
        return -loss.mean()  # [1]
def train_posi_embed(netData: Data, emb_dim: int, epoches=15):
    '''in netData
        out, Embedding net & loss recorder'''
    W, K = 4, 4
    #batch_size = 5
    lr = 0.01
    embeddingNet = PosiEmbedding(netData.num_nodes, emb_dim)
    lossfunc = NegativeSamplingLoss()
    optimizer = torch.optim.Adam([p for p in embeddingNet.parameters() if p.requires_grad==True], lr=lr)
    lossRecorder = Accumulator(1)
    for epoch in range(epoches):
        for formbatch in range(0, netData.num_nodes):  # _是每个batch的开始id
            optimizer.zero_grad()
            # 直接batch = 1 吧 orzzz
            center = []  # [batch, 1]
            neighbor = []  # [batch, num_neighbors]
            negs = []    # [batch, num_negs]
            center.append([formbatch])
            curneighbor_l = [formbatch]  # 至少有自己，让孤立节点不至于出现训练不了的情况
            for i in range(netData.num_edges):  # 寻找邻居节点
                if int(netData.edge_index[0][i]) == formbatch:
                    curneighbor_l.append(netData.edge_index[1][i])
            cur_neg_l = [neg for neg in range(netData.num_nodes) if neg not in curneighbor_l]  # 不与其相邻的都是负样本
            if len(curneighbor_l) > 4:
                random.shuffle(curneighbor_l)
                curneighbor_l = curneighbor_l[:4]
            if len(cur_neg_l) > 4:
                random.shuffle(cur_neg_l)
                cur_neg_l = cur_neg_l[:4]
            neighbor.append(curneighbor_l)
            negs.append(cur_neg_l)
            center, neighbor, negs = torch.LongTensor(center), torch.LongTensor(neighbor), torch.LongTensor(negs)  # 形成了batch的id
            center, neighbor, negs = embeddingNet.forward_input(center), embeddingNet.forward_target(neighbor), embeddingNet.forward_target(negs)
            l = lossfunc(center, neighbor, negs)
            l.backward()
            optimizer.step()
            lossRecorder.add(l.detach().numpy())
        lossRecorder.clear()
    #print(lossRecorder.backup)

    return embeddingNet, lossRecorder

if __name__ == '__main__':
    df = read_GithubData_mycombine_outfielder()
    #gk = df['']
    
