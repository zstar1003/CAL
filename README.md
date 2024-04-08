# HUS-ADC
HUS-ADC: A Two-Phase Active Learning Strategy for Object Detection


# 脚本功能
get_conf_pic.py：根据置信度选择置信度最低的n张图片
get_defdomain_pic.py：根据源域和目标域差异选取最大的n张图片
get_divergence_pic.py：选择委员会分歧度最大的n张图片
get_entropy_pic.py：选择熵最大的n张图片
get_random_pic.py：随机选择n张图片
get_cluster_pic.py: 根据目标域聚类结果选择n张图片
get_classbalence_pic.py: 根据目标域类别分布选择n张图片