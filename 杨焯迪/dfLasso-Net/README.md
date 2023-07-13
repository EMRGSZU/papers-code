该文件包含了dfLasso-Net在三个数据集上的测试代码，包括：
(1)DSA dataset
(2)Skoda dataset
(3)PAMAP2 dataset

其中dataprocess.py为数据集处理过程，train.py开始训练，model.py包含网络模型代码。
该代码中的dfLasso-Net为嵌入式特征选择方法，若要调整为过滤式特征选择方法只需在得到特征子集以后额外输入到分类器测试即可。
网络的整体结构图可以查看dfLasso-Net.vsdx。
