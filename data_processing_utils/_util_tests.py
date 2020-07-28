import pandas as pd
import numpy as np
from data_processing_utils._processing_funcs import ResultProcessing

# test get network function
result = ResultProcessing()
result.load_models_from_file_path("../dataset/lgp_acc.pkl")
data = pd.read_csv('../assets/sample_data/sample_alzheimer_vs_normal.csv')
X, y = ResultProcessing.read_dataset_X_y(data)
names = ResultProcessing.read_dataset_names(data)
result.calculate_featureList_and_calcvariableList()

df, node_size_dic = result.get_network_data(names, 0.03)
print(node_size_dic)
print(df)
top = np.unique(df[['f1', 'f2']].values)
print(top)
# for index, row in df.iterrows():
#     print(df['source'][index])

# prog_index, acc_scores =  result.get_accuracy_given_length(1)

# index = result.get_index_of_models_given_feature_and_length(105, 3)
# print(index)
# for i in index:
#     print(result.model_list[i].bestEffProgStr_)

# print(result.model_list[205].bestEffProgStr_)
# s = result.convert_program_str_repr(result.model_list[205])

# co_matrix, featureIndex = result.get_feature_co_occurences_matrix(2)
# hover_text = []
# for yi, yy in enumerate(featureIndex):
#     hover_text.append([])
#     for xi, xx in enumerate(featureIndex):
#         hover_text[-1].append('X: {}<br />Y: {}<br />Count: {}'.format(xx, yy, co_matrix[xi,yi]))
# print(hover_text)