# import re

# text = """Based on the provided output, it seems that a Random Forest Classifier with the following hyperparameters is giving the best performance (lowest error) on your dataset:

# - max_depth: 10
# - max_features: 0.86
# - min_impurity_decrease: 0
# - min_samples_leaf: 0.05
# - min_samples_split: 0.37
# - min_weight_fraction_leaf: 0.25

# The performance (error) of this model is 1.01775"""


# text = "max_depth: 7, max_features: 0.62, min_impurity_decrease: 0, min_samples_leaf: 0.03, min_samples_split: 0.41, min_weight_fraction_leaf: 0.15"


# text ="""max_depth: 11, max_features: 0.77, min_impurity_decrease: 0, min_samples_leaf: 0.03, min_samples_split: 0.42, min_weight_fraction_leaf: 0.25

# This configuration is recommended based on the provided performances and hyperparameter configurations, aiming for a target performance of 0.999487 with the highest possible precision within the allowed ranges.
# """

# # 使用正则表达式找出所有匹配的参数和值
# matches = re.findall(r'(\w+): ([\d.]+)', text)

# # 格式化提取的信息
# formatted_string = ', '.join([f"{match[0]}: {match[1]}" for match in matches])

# print(formatted_string)



import re

str1 = "## 0.4291 0.6767"
str2 = "## 0.4291 ##"

pattern = r"\d+\.\d+"

match1 = re.findall(pattern, str1)
match2 = re.findall(pattern, str2)

if match1:
    print("Found number in str1:", match1)
    print(len(match1))
else:
    print("No number found in str1.")

if match2:
    print("Found number in str2:", match2)
else:
    print("No number found in str2.")
