# Traffic-speed-prediction

## 注意事项

<font face="微软雅黑" >需要注意的是，需要根据requirements.txt文件中指示的包进行安装，才能正常的运行程序！！！</font>
  
>* 首先，使用conda创建一个虚拟环境，如‘conda create traffic_speed’；  
> * 激活环境，conda activate traffic_speed；  
> * 安装环境，需要安装的环境已经添加在requirements.txt中，可以用conda安装，也可以使用pip安装，如：conda install tensorflow==1.12.0；  
> * 如果安装的是最新的tensorflow环境，也没问题，tensorflow的包按照以下方式进行导入即可：import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()；  
> * 点击 run_train.py文件即可运行代码。
> * 需要注意的是，我们在tensorflow的1.12和1.14版本环境中都可以运行
---

## Experimental Results
### HA (Multi-steps)
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |
| MAE            |6.185242 |6.185242 |6.185242 |6.185242 |6.185242 |6.185242 |
| RMSE           |10.139710|10.139710|10.139710|10.139710|10.139710|10.139710|
| MAPE           |0.091109 |0.091109 |0.091109 |0.091109 |0.091109 |0.091109 |
| R              |0.941854 |0.941854 |0.941854 |0.941854 |0.941854 |0.941854 |
| R<sup>2</sup>  |0.878433 |0.878433 |0.878433 |0.878433 |0.878433 |0.878433 | 
---
### ARIMA (Multi-steps)
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |
| MAE            |5.883667 |6.046000 |6.104512 |6.374540 |6.360256 |6.168883 |
| RMSE           |9.429440 |9.798609 |9.769539 |10.209594|10.030069|9.769304 |
| MAPE           |0.102238 |0.159162 |0.130914 |0.110670 |0.129478 |0.131120 |
| R              |0.941224 |0.937721 |0.937789 |0.930736 |0.933817 |0.937160 |
| R<sup>2</sup>  |0.885876 |0.879284 |0.879403 |0.866113 |0.871992 |0.878266 |
---
### SVM (Multi-steps)
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |
| MAE            |5.622419 |5.694858 |5.636928 |5.902129 |5.898720 |5.929169 |
| RMSE           |9.541193 |9.459830 |9.366811 |9.714313 |9.744125 |9.761787 |
| MAPE           |0.085023 |0.086360 |0.083737 |0.089644 |0.087872 |0.090589 |
| R              |0.948368 |0.949696 |0.950375 |0.947028 |0.946775 |0.946549 |
| R<sup>2</sup>  |0.892623 |0.894429 |0.896641 |0.888217 |0.887459 |0.887585 | 
---
### LSTM (Multi-steps)
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |
| MAE            |5.812265 |6.145748 |5.782633 |5.955329 |5.791204 |5.508455 |
| RMSE           |9.411718 |9.885983 |9.367372 |9.556509 |9.442872 |9.024291 |
| MAPE           |0.087702 |0.093565 |0.088276 |0.089859 |0.087029 |0.080138 |
| R              |0.951281 |0.944342 |0.949958 |0.947607 |0.949351 |0.953303 |
| R<sup>2</sup>  |0.896236 |0.880975 |0.893135 |0.888133 |0.892056 |0.900712 | 
---
### Bi-LSTM (Multi-steps)
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |
| MAE            |5.818706 |6.149838 |5.787297 |5.965189 |5.796960 |5.516990 |
| RMSE           |9.410817 |9.885696 |9.368603 |9.563658 |9.438372 |9.030807 |
| MAPE           |0.088159 |0.094087 |0.088030 |0.089670 |0.087124 |0.080557 |
| R              |0.951324 |0.944394 |0.949980 |0.947582 |0.949448 |0.953280 |
| R<sup>2</sup>  |0.896564 |0.881313 |0.893332 |0.888252 |0.892393 |0.900803 |
---
### FI-RNNs (Multi-steps)
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |
| MAE            |5.802633 |6.127445 |5.771666 |5.938240 |5.783873 |5.500322 |
| RMSE           |9.394858 |9.867163 |9.354874 |9.530777 |9.434043 |9.020741 |
| MAPE           |0.086731 |0.092550 |0.086666 |0.088448 |0.085913 |0.079748 |
| R              |0.951490 |0.944613 |0.950145 |0.947960 |0.949502 |0.953402 |
| R<sup>2</sup>  |0.895826 |0.880546 |0.892533 |0.887802 |0.891259 |0.899899 |  
---
### PSPNN (Multi-steps)
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |
| MAE            |5.557420 |5.849365 |5.528135 |5.720642 |5.596671 |5.404817 |
| RMSE           |9.044259 |9.480751 |9.041988 |9.212497 |9.138348 |8.872272 |
| MAPE           |0.089233 |0.093376 |0.090259 |0.091873 |0.091068 |0.089050 |
| R              |0.955093 |0.948999 |0.953482 |0.951480 |0.952706 |0.955044 |
| R<sup>2</sup>  |0.903878 |0.891049 |0.901076 |0.896809 |0.898577 |0.903503 | 
---
### MDL (Multi-steps)
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |
| MAE            |5.499502 |5.849312 |5.514862 |5.722308 |5.716724 |5.666359 |
| RMSE           |8.918145 |9.403159 |8.974397 |9.152807 |9.313859 |9.346188 |
| MAPE           |0.090207 |0.094943 |0.091913 |0.093620 |0.094586 |0.095070 |
| R              |0.956450 |0.950058 |0.954299 |0.952301 |0.950966 |0.950197 |
| R<sup>2</sup>  |0.907563 |0.893937 |0.903611 |0.899380 |0.896112 |0.895045 |  
---
### T-GCN (Multi-steps)  
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |
| MAE            |5.617051 |5.900797 |5.663260 |5.824718 |5.901005 |5.767865 |
| RMSE           |8.976242 |9.389217 |9.029338 |9.197953 |9.463521 |9.387994 |
| MAPE           |0.092815 |0.100017 |0.102152 |0.108812 |0.103518 |0.111413 |
| R              |0.955867 |0.950061 |0.953587 |0.951662 |0.949146 |0.949376 |
| R<sup>2</sup>  |0.904803 |0.892950 |0.901571 |0.897469 |0.891171 |0.892146 | 
---
### AST-GAT (Multi-steps)
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |
| MAE            |5.170596 |5.672411 |5.411912 |5.405243 |5.345344 |4.996449 |
| RMSE           |8.489692 |9.265172 |8.827237 |8.858307 |8.952828 |8.362807 |
| MAPE           |0.089477 |0.213028 |0.065361 |-0.035309|0.075862 |0.049634 |
| R              |0.960515 |0.951535 |0.955691 |0.955131 |0.954533 |0.959999 |
| R<sup>2</sup>  |0.915572 |0.897554 |0.906764 |0.905433 |0.902892 |0.915488 | 
---
### GMAN (Multi-steps)  
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |
| MAE            |5.276713 |5.295712 |5.180604 |5.320412 |5.266498 |5.239795 |
| RMSE           |8.806952 |8.931492 |8.597085 |8.832887 |8.779651 |8.699286 |
| MAPE           |0.150773 |0.068655 |0.009746 |0.116505 |0.108108 |0.077505 |
| R              |0.955977 |0.954947 |0.958259 |0.956372 |0.956673 |0.957122 |
| R<sup>2</sup>  |0.907923 |0.905585 |0.912525 |0.908170 |0.908718 |0.910450 | 
---
### GMAN_1 (without decoder on spatiotemporal attention)  
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |
| MAE            |5.250782 |5.267231 |5.169138 |5.317462 |5.229249 |5.223629 |
| RMSE           |8.787853 |8.921577 |8.601191 |8.836030 |8.755781 |8.692222 |
| MAPE           |0.079309 |0.079563 |0.077797 |0.079259 |0.079230 |0.078870 |
| R              |0.956120 |0.954998 |0.958175 |0.956284 |0.956876 |0.957135 |
| R<sup>2</sup>  |0.907388 |0.904933 |0.911455 |0.907111 |0.908335 |0.909635 | 
---
### STGIN (Multi-steps) 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |
| MAE            |5.099863 |5.443436 |5.133026 |5.317708 |5.158664 |4.891572 |
| RMSE           |8.510213 |9.033206 |8.581541 |8.814219 |8.697461 |8.408962 |
| MAPE           |0.081441 |0.086490 |0.082539 |0.083498 |0.082179 |0.078129 |
| R              |0.960282 |0.953856 |0.958171 |0.955643 |0.957188 |0.959637 |
| R<sup>2</sup>  |0.915884 |0.902442 |0.912126 |0.906875 |0.909585 |0.915334 |  
---
### STGIN_1 (Multi-steps) without semantic transformation
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |
| MAE            |5.126470 |5.448768 |5.128695 |5.326809 |5.191423 |4.907069 |
| RMSE           |8.520419 |9.028870 |8.536077 |8.805729 |8.724048 |8.418258 |
| MAPE           |0.077254 |0.081946 |0.077594 |0.079762 |0.077379 |0.071435 |
| R              |0.960223 |0.953855 |0.958595 |0.955656 |0.956895 |0.959483 |
| R<sup>2</sup>  |0.913848 |0.900768 |0.911281 |0.905346 |0.907142 |0.913316 | 
---
### STGIN_2 (Multi-steps) uses an encoder of GMAN to replace the of ST-Block STGIN
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |
| MAE            |5.108122 |5.452237 |5.133146 |5.341251 |5.186359 |4.964654 |
| RMSE           |8.553639 |9.068058 |8.613500 |8.856792 |8.794576 |8.513375 |
| MAPE           |0.076658 |0.081614 |0.077310 |0.079311 |0.076734 |0.071927 |
| R              |0.959957 |0.953760 |0.957960 |0.955410 |0.956393 |0.958854 |
| R<sup>2</sup>  |0.915391 |0.901909 |0.911698 |0.906351 |0.907654 |0.913126 | 
---
### STGIN_3 (Multi-steps) dynamic step-by-step method to replace the generative inference of STGIN
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |
| MAE            |5.143526 |5.487834 |5.158540 |5.342705 |5.190592 |4.952848 |
| RMSE           |8.563437 |9.080045 |8.614929 |8.815262 |8.719995 |8.491023 |
| MAPE           |0.084558 |0.088534 |0.086561 |0.087878 |0.085525 |0.084191 |
| R              |0.959812 |0.953493 |0.957831 |0.955652 |0.956994 |0.958873 |
| R<sup>2</sup>  |0.914000 |0.900800 |0.910693 |0.906278 |0.908481 |0.912903 | 
---
## 代码更正
##### 如果你想保证模型的精准度更高，请将 bridge.py 中的代码，改为以下代码 (block set to 1， and input length set to 12)
* 新版STGIN映射到STGIN_4权重weights/参数上，以此类推，STGIN_3映射到STGIN_7权重weights/参数上，  
            # Blocks  
                for i in range(self.num_blocks):  
                        with tf.variable_scope("num_blocks_{}".format(i)):  
                            # Multihead Attention  
                            X_Q = multihead_attention(queries=X_Q, # future time steps  
                                                   keys=X_P,    # historical time steps  
                                                 values= X,   # historical inputs
                                                   num_units=self.hidden_units,  
                                                  num_heads= self.num_heads, # self.num_heads  
                                                 dropout_rate=self.dropout_rate,  
                                                    is_training=self.is_training)  
                         # Feed Forward  
                         X_Q = feedforward(X_Q, num_units=[4 * self.hidden_units, self.hidden_units])  
             X = tf.reshape(X_Q,shape=[-1, self.site_num, self.output_length, self.hidden_units])  
---