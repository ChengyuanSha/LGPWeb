import pandas as pd
import numpy as np
from data_processing_utils._processing_funcs import ResultProcessing


result = ResultProcessing()
result.load_models_from_file_path("../dataset/lgp_random_AD_vs_Normal.pkl")
data = pd.read_csv('../assets/sample_data/sample_alzheimer_vs_normal.csv')
X, y = ResultProcessing.read_dataset_X_y(data)
names = ResultProcessing.read_dataset_names(data)
result.calculate_featureList_and_calcvariableList()
# test get network function
# df, node_size_dic = result.get_network_data(names, 0.03, 'dUMP')
# print(node_size_dic)
# print(df.values)
#
# aaa = df.loc[(df['f1'] == 'dUMP') | (df['f2'] == 'dUMP')]
#
# others = np.unique(aaa[['f1', 'f2']].values)
# others = others[others != 'dUMP']
# aaa2 = df.loc[(df['f1'].isin(others)) & (df['f2'].isin(others)) ]
# aaa = aaa.append(aaa2, ignore_index=True)
# print("dd")
# end get network function


# for index, row in df.iterrows():
#     print(df['source'][index])

# prog_index, acc_scores =  result.get_accuracy_given_length(1)

# index = result.get_index_of_models_given_feature_and_length(105, 3)
# print(index)
# for i in index:
#     print(result.model_list[i].bestEffProgStr_)

print(result.model_list[571].bestEffProgStr_)
s = result.convert_program_str_repr(result.model_list[571], names)
print(s)
# -----test get_feature_co_occurences_matrix function----

# dump = 105
# co_matrix, featureIndex = result.get_feature_co_occurences_matrix('All')
# featureIndex = np.array(featureIndex)
# if dump in featureIndex:
#     f_index = np.where(featureIndex == dump)
#     f_row = np.array(co_matrix[f_index])
#     nonzero_index = np.where(co_matrix[f_index]>0)
#     cooccurring_times = f_row[nonzero_index]
#     cooccurring_features = featureIndex[nonzero_index[1]]
#
# a , b = result.get_cooccurrence_info_given_feature(105)
# print(a, b)
# hover_text = []
# for yi, yy in enumerate(featureIndex):
#     hover_text.append([])
#     for xi, xx in enumerate(featureIndex):
#         hover_text[-1].append('X: {}<br />Y: {}<br />Count: {}'.format(xx, yy, co_matrix[xi,yi]))
# print(hover_text)