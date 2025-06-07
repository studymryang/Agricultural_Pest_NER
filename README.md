# Agricultural_Pest_NER
农业病虫害命名实体识别</br>
1.在文件中model_hub中需要去Hugging Face中去下载chinese-bert-wwm-ext模型pytorch的bin文件放在文件中即可，下载好后开箱即用 </br>
![image](https://github.com/user-attachments/assets/59ae34a0-2c22-49c0-a9f2-4cb61119fcbc) </br>
2.model里提供了一下模型 </br>
![image](https://github.com/user-attachments/assets/93ba6673-d910-41f1-9737-e3115f3d98ca) </br>
3.glyph文件是字形向量</br>
在model.py也提供了bert融合字形向量的命名实体识别模型，但是效果不是很好。
