 Emotion_Analysis
 =========
 微博情感分析
 ---------
* code（实验代码）<br>
	* automatic_vec.py：深度学习特征(word2vec学习到的词向量)进行训练，其中包含logistic regression、knn、cnn、rnn分类器的实验结果。<br>
	* manual_vec.py：人工选取的特征进行训练，其中包含logistic regression、svm分类器的实验结果。<br>
	* sub_obj.py:	主观句和客观句分类。<br>
* dataset（数据集）<br>
	* word2vec训练集：word2vec训练语料<br>
	* 情感词和停用词表。<br>
	* 正负样例：pos.txt   neg.txt各15000条褒贬句（酒店评论和书籍评论等）。<br>
	* 主观和客观样例：sub.txt   obj.txt各500条数据<br>
