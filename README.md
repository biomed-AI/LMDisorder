# Introduction
LMDisorder is a fast and accurate protein disorder predictor that employed embedding generated by unsupervised pretrained language models as features.We showed that LMDisorder essentially surpassed the single-sequence-based methods by more than 6.0% and 18.0% on AUROC in two independent test sets, respectively. Furthermore, LMDisor-der showed equivalent or even better performance than the state-of-the-art profile-based technique SPOT-Disorder2.
![image](https://github.com/songyidong-true/LMDisorder/blob/main/image/LMDisorder_architecture.png)
# System requirement
python 3.7.9  
numpy 1.19.1  
pandas 1.1.0  
pytorch 1.10.0  
sentencepiece 0.1.96  
transformers 4.18.0  
tqdm 4.48.2  
# Pretrained language model
You need to prepare the pretrained language model ProtTrans to run LMDisorder:
Download the pretrained ProtT5-XL-UniRef50 model ([guide](https://github.com/agemagician/ProtTrans)). # ~ 11.3 GB (download: 5.3 GB)
# Run LMDisorder for prediction
Simply run:  
```
python LMDisorder_predict.py --fasta ./example/demo.fasta --device 'cpu' --model_path ./model/model.pkl
```
And the prediction results will be saved in
```
./example/result
```
We also provide the corresponding canonical prediction results in ```./example/demo_result``` for your reference.
# Dataset and model
We provide the datasets and the trained LMDisorder models here for those interested in reproducing our paper. The datasets used in this study are stored in ```./datasets/```.
The trained LMDisorder models can be found under ```./model/```.
# Citation and contact
Citation: 
```bibtex
@article{10.1093/bib/bbad173,
    author = {Song, Yidong and Yuan, Qianmu and Chen, Sheng and Chen, Ken and Zhou, Yaoqi and Yang, Yuedong},
    title = "{Fast and accurate protein intrinsic disorder prediction by using a pretrained language model}",
    journal = {Briefings in Bioinformatics},
    year = {2023},
    month = {05},
    issn = {1477-4054},
    doi = {10.1093/bib/bbad173},
    url = {https://doi.org/10.1093/bib/bbad173}    
}
```

Contact:

Yidong Song (songyd6@mail2.sysu.edu.cn)  
Yuedong Yang (yangyd25@mail.sysu.edu.cn)
