# Traffic-speed-prediction

We have added all the baseline models‘ results and codes to our GitHub page, including ST-GRAT, GMAN, DCRNN, T-GCN, Etc! Note, we have added the ST-GRAT and DCRNN results to our paper even though the prediction precisions of DCRNN and ST-GRAT (implemented by PyTorch, torch==1.6.0) are further lower than GMAN, but we have gave the results on the dataset as the experiment shows. Our task is using the 12 historical time steps to prediction the future 12 target time steps traffic speed of highway, and our time granularity is 15 minutes. This proposed highway traffic speed prediction model, i.e, STGIN, has been accepted and published in the Transportation Research Part C. We have uploaded and upgraded the source codes, such as adding the last week's speed pattern in the predicting phase. If you have any questions, don't hesitate to connect us, thanks！

## WHAT SHOULD WE PAY ATTENTION TO FOCUS ON THE RUNNING ENVIRONMENT?

<font face="微软雅黑" >Note that we need to install the right packages to guarantee the model runs according to the file requirements.txt！！！</font>
  
>* first, please use the conda tool to create a virtual environment, such as ‘conda create traffic speed’;  
> * second, active the environment, and conda activate traffic speed;   
> * third, build environment, the required environments have been added in the file named requirements.txt; you can use conda as an auxiliary tool to install or pip, e.g., conda install tensorflow==1.13.1;    
> * if you have installed the last TensorFlow version, it’s okay; import tensorflow.compat.v1 as tf and tf.disable_v2_behavior();    
> * finally, please click the run_train.py file; the codes are then running;  
> * Note that our TensorFlow version is 1.14.1 and can also be operated on the 2.0. version.  
---

## DATA DESCRIPTION  
> The traffic speed data used in this study is provided by the ETC intelligent monitoring sensors at the gantries and the toll stations of the highway in Yinchuan City, Ningxia Province, China. The 66 ETC intelligent monitoring sensors record the vehicle driving data in real-time, including 13 highway toll stations (each toll station contains an entrance and exit) and 40 highway gantries. Therefore, these monitoring sensors divide the highway network into 108 road segments. The traffic speed of each road segment is measured at a certain frequency, such that one sample is measured every 15 minutes, and therefore the time series form of the traffic speed is obtained. In addition, the traffic speed data also includes the other two factors, timestamps and road segment index. Because of traffic speed heterogeneity on different types of road segments, the traffic speed is divided into three types, from entrance toll to gantry, called ETTG; gantry to gantry, called GTG; and gantry to exit toll, called GTET. The time span is from June 1, 2021, to August 31, 2021. The road segment index does not change over time, and there are 108 road segments in total, that is, 108 indexes. In the experiment, 70% of the data are used as the training set, 10% of the data are used as the validation set, and the remaining 20% are considered as the test set ([dataset link](https://github.com/zouguojian/STGIN/tree/main/data)).
---

## EXPERIMENTAL SETTING  

> We applied the grid search in the proposed STGIN method to find the optimal model on the validation dataset. Especially among all candidate hyperparameter selections, every possibility is tried through loop traversal, and the hyperparameter group with the best performance on the validation dataset is selected as the final result. Note for these continuous hyperparameter values, sample at equal intervals. For each hyperparameter group, the optimal parameters of the proposed STGIN model and baseline techniques are determined during the training process with minimal MAE on the validation set, and specific processing follows, 

 > In the experiment, the maximum number of epochs is 100; and the batch size is 32, which divides the training set into 183 iterations in a single epoch. Updating the model’s parameters via backpropagation with a batch of data is called one iteration.  Specifically, we evaluate the prediction model on the validation set after one epoch. If the MAE on the validation set is improved, the model parameters are updated and recorded to replace the last one saved. In addition, when the forecasting performance of the prediction model on the validation set is optimal, the training process ends after many parameter adjustments and experiments. We use an early-stop mechanism in all experiments, and the number of early-stop epochs is set to 10, defined as patience. The early-stop mechanism means the training stops early if the MAE on the validation set is not decreased under the patience before the maximum number of epochs. Finally, the prediction result is obtained by iterating all the samples in the test set. We set the target time steps $\rm Q$ and historical time steps $\rm P$ to 12, respectively, representing the time span is 360 minutes. 
---
## METRICS

> In order to evaluate the prediction performance of the MT-STGIN model, three metrics are used to determine the difference between the observed values $\rm Y$ and the predicted values $\rm \hat{Y}$ : the root mean square error (RMSE), mean absolute error (MAE), and mean absolute percentage error (MAPE). Note that low MAE, RMSE, and MAPE values indicate a more accurate prediction performance.  

> 1.  MAE (Mean Absolute Error):


# latex形式
![](https://latex.codecogs.com/svg.image?MAE=\frac{1}{N}\sum_{1}^{N}\left|\hat{Y}-Y\right|)


${\rm MAE}=\frac{1}{\rm N} \sum_{i=1}^{\rm N}\left|\hat{\rm Y}_{i}-{\rm Y}_{i}\right|$

> 2. RMSE (Root Mean Square Error):

$$
{\rm RMSE} =\sqrt{\frac{1}{\rm N} \sum_{i=1}^{\rm N}\left(\hat{\rm Y}_{i}-{\rm Y}_{i}\right)^{2}}
$$

> 3. MAPE (Mean Absolute Percentage Error):

$$
{\rm MAPE}=\frac{100 \%}{\rm N} \sum_{i=1}^{\rm N}\left|\frac{\hat{\rm Y}_{i}-{\rm Y}_{i}}{{\rm Y}_{i}}\right|
$$


## PREDICTING PERFORMANCE COMPARISON 

> Performance comparison of different approaches for long-term highway traffic speed prediction  

<div align=center><img src ="https://github.com/zouguojian/STGIN/blob/main/figs/1.png" width = "1200" height="370"/></div>

## INFLUENCE OF EACH COMPONENT

> Performance of the different time steps prediction for distinguished variants  

<div align=center><img src ="https://github.com/zouguojian/STGIN/blob/main/figs/2.png" width = "800" height="230"/></div>

## COMPUTATION COST

> Computation cost during the training and inference phases (* means the model train one time on the whole training set) 

<div align=center><img src ="https://github.com/zouguojian/STGIN/blob/main/figs/3.png" width = "800" height="350"/></div>
