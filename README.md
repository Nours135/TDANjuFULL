# TDANjuFULL
## 环境
python = 3.9 <br/>
主要使用的库: kmapper, sklearn, pytorch, pytorch geometry, gitto-tda

## 文件结构
### 数据 & 描述统计
数据1：2021-2022赛季五大联赛数据<br/>
源数据，[datacleaning\modified_data.csv](datacleaning/modified_data.csv) <br/>
字段解释，[datacleaning/colsStandard.xlsx](datacleaning/colsStandard.xlsx) <br/>
描述统计结果：[datacleaning/Descriptive.docx](datacleaning/Descriptive_statistics.doc) & [描述统计.xlsx](datacleaning/%E6%8F%8F%E8%BF%B0%E7%BB%9F%E8%AE%A1.xlsx) <br/>
一些统计图，箱型图，散点图，直方图，datacleaning/boxplot, datacleaning\histogram, datacleaning\scatter
数据2：2017-2018至2020-2021赛季数据 <br/>
源数据，[DLdata/outfielder_combined_from_raw.xlsx](DLdata/outfielder_combined_from_raw.xlsx) <br/>
使用随机森林回归填补缺失值后（代码在utils.py中）的数据（缺失值最多的字段是10000条含有1000条缺失）：[DLdata/outfielder_combined_from_raw_fillna.csv](DLdata/outfielder_combined_from_raw_fillna.csv)<br/>
关联的球队表现数据：[DLdata/SquadPerformance.xlsx](DLdata/SquadPerformance.xlsx), [DLdata/SquadPerformance2021.xlsx](DLdata/SquadPerformance2021.xlsx)
字段解释，暂无<br/>
描述统计结果：[\[datacleaning/Descriptive.docx\](datacleaning/Descriptive_statistics_2.doc)](datacleaning/Descriptive_statistics_2.doc) & [datacleaning/描述统计 2.xlsx](<datacleaning/描述统计 2.xlsx>) <br/>

### 代码
数据集1的mapper生成，运行[mapper/mapper.ipynb](mapper/mapper.ipynb)，或者运行[Tda_basic_flare_vacancy.ipynb](Tda_basic_flare_vacancy.ipynb)，在初始化类的时候将save参数设置为True
