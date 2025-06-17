# Agricultural_Pest_NER
此任务是农业病虫害命名实体识别，数据集的标注格式采用bio的形式标注</br>
![image](https://github.com/user-attachments/assets/4e1f4b08-cc65-4a5f-8bab-034e190df9d8)</br>
如果你标注的bio格式是如下格式的话，需要转换为上述格式,这里提供给转换格式代码 <a href=https://blog.csdn.net/weixin_45401410/article/details/138368177?spm=1001.2014.3001.5502>转换数据集地址链接</a> </br>
![image](https://github.com/user-attachments/assets/313f9a6b-004a-4458-9654-b7f63bb28eec)

## 这里给大家提供了6中不同的命名实体识别模型
### BERT-BiLSTM-CRF
### BERT-BiGRU-CRF
### BiGRU-CRF
### BiLSTM-CRF
### BERT-CRF
### 还用一种融合了字形，但是效果不好，具体可以自己去实验

## 运行流程
### 1.你要去 Hugging Face 去下载bert配置文件放到model_hub下
### 2.运行main.py 在main.py可以通过注释来选择不同的模型
