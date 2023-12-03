# SNA_project read me 

## Required datasets
The dataset for this project is confidential and cannot be provided as part of project submission.

# Instructions
## Model hyperparameter search 
1. Install Python 3.10 (Requires python 3.10 for Ray hyperparameter search)
2. `pip install -r requirements.txt`

Note: The requirements.txt might not work for generating the graphs (i.e. files inside SNA_create_graph folder). We have included an alternate requirements file (requirements_create_graph.txt)

## Code instructions
The code for each model created are stored in separate folders and the structure for the same is below, with a short description for each folder and file.
```
├── SNA_create_graph
│   ├── plots
│   ├── 00_process_wego_apc.py (Python file for getting APC bus transit data and pre-processing it)
│   ├── 01_merge_traffic_data.py (Python file for merging APC bus transit data and inrix traffic data)
│   ├── 02_data_preprocess_add_weather.py (Python file for merging APC bus transit data and weather data)
│   ├── 03_create_new_graph_data_dynamic_graph_static_nodes.py (Python file for creating the static graph - 1hr time window)
│   ├── 03_create_new_graph_data_no_4th_bin_1_hr_time_window.py (Python file for creating the dynamic graph - 2hr time window)
│   ├── 03_create_new_graph_data_no_4th_bin_2_hr_time_window.py (Python file for creating the dynamic graph - 2hr time window)
Note: In SNA create graph folder above all above file a notebook file is also present
├── SNA_Project_dynamic_graph
│   ├── exp_results (Folder with history of training and validation results for each epoch)
│   ├── results (Folder with prediction results)
│   ├── 04_train_dynamic_1_hr_GConvGRU.py (Python file for training dynamic graph model and generating baseline results - 1 hour time window)
│   ├── 04_train_dynamic_1_hr_GConvGRU.ipynb (Notebook for above file)
│   ├── 04_train_dynamic_2_hr_GConvGRU.py (Python file for training dynamic graph model and generating baseline results - 2 hour time window)
│   ├── 04_train_dynamic_2_hr_GConvGRU.ipynb (Notebook for above file)
│   ├── 05_baseline__dynamic_graph_1_hr.py (Python file for training baselines and generating baseline results)
│   ├── 05_baseline__dynamic_graph_1_hr.ipynb (Notebook for above file)
│   ├── get_results.ipynb (Notebook for generating Confusion matrix and accuracy metrics.)
├── SNA_Project_static_graph
│   ├── exp_results (Folder with history of training and validation results for each epoch)
│   ├── results  (Folder with prediction results)
│   ├── 04_train_static_1_hr_GConvGRU_static.py (Python file for training static graph model and generating baseline results - 1 hour time window)
│   ├── 04_train_static_1_hr_GConvGRU_static.ipynb (Notebook for above file)
```

# Input Features
    * precipitation_intensity: Numerical;
    * temperature: Numerical;
    * humidity: Numerical;
    * delay: Numerical;
    * average_speed: Numerical; 
    * extreme_congestion: Numerical;
    * dayofweek: Categorical;
    * month: Categorical;
    * year: Categorical;
    * time_window: Categorical.

# Output: `load`
* Bins based on vehicle capacity, based on Transit App.
    * 0-33% of max capacity
    * 33-66% of max capacity
    * 66-100% of max capacity
    * 100%+ of max capacity
    * A 5th bin exists for static graph model that indicates that no bus passes through the stop in that time window