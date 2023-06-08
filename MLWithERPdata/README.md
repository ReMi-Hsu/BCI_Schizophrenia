'statistic_ERPdata.ipynb' is to statistic ERP Data

'randomforset_forERPdata.ipynb' is to preprocess ERP Data and using some ML method (SVM, randomForest) to do classification. 

![](https://github.com/ReMi-Hsu/BCI_Schizophrenia/blob/main/MLWithERPdata/N1supp.jpg)

It shows that HC and SZ exists a gap of N1 suppression, so maybe we can easily use the feature of N1 suppression to classify.
- the shape of each data is (2, )
    - one is readiness potential (RP), the other is N1 suppression (N1supp) 
        - the processing detail can see randomforset_forERPdata.ipynb
        
However, ERP data has 81 subjects, and each subject has only 1 trails
- That means the number of data in ERP Dataset is only 81.
    - 64 for training, 17 for testing 

The number of training data is 64, which is less. Thus we use machine learning method to classify.
- Using methods:
    - Random forests: good way to classify small training samples.
    - SVM

- Result 

![](https://github.com/ReMi-Hsu/BCI_Schizophrenia/blob/main/MLWithERPdata/result.jpg)

