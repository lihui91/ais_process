### dataset and preprocessing

#### Execution order:

##### Training set & validation set： 

csv2pkl.py --> dataset_preprocessing.py

##### Test Set：

1. dataset_preprocessing_test_stage_1.py
2. inject anomaly
3. dataset_preprocessing_test_stage_2.py

#### anomaly injection
shift deviate: inject_type_1_deviate.py
abnormal heading: inject_type_2_abnormal_heading.py
abnormal speed: inject_type_3_sog_noise.py