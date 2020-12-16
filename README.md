# cmpt-726-project

##### Author: Bowen Song,  Suohong Liu, Weijiang Pan

#### Set Environment

> ```shell
> #create conda environment with yml file
> conda env create -f environment.yml
> #activae environment
> source activate project_726
> ```

#### Dataset:

> * sign language MNIST: https://www.kaggle.com/datamunge/sign-language-mnist
>
> * ```shell
>   cmpt-726-project
>   ├── data (bigger dataset with 17000 samples)
>   |   └── test.csv 
>   |   └── train.csv 
>   │   └── data2 (small dataset with 3000 samples )
>   │       └── test.csv
>   |       └── train.csv
>   ```

#### Suohong Liu

> ```shell
> cmpt-726-project
> ├── gesturePredict.ipynb
> |
> ├── models
> |   └── model_self.model
> |
> ├── data
> |   └── demo.mp4
> |   └── test.csv
> |   └── train.csv
> │   └── data2
> │       └── test.csv
> |       └── train.csv
> └── code
> |   └── data_collection
> |       └── collectImage.ipynb
> |       └── imageToCsv.ipynb
> |   └── model
> |       └── model_simple.ipynb
> |       └── model_2.ipynb
> ```
>
#### Weijiang Pan
> ```shell
> cmpt-726-project
> |
> ├── models
> |   └── resnet.model
> |
> └── code
> |   └── model
> |       └── sing_language_resnet50.ipynb
> |       
> ```

> `gesturePredict.ipynb` : real-time gesture recognize, pls do the gesture in ROI (region of interest) box.
>
> `model_self.model`: trained data from `model_2.ipynb`
>
> `demo.mp4`: a demo of real-time ASL recognize using `gesturePredict.ipynb`
>
> `test.csv`: testing data
>
> `train.csv `: training data
>
> `collectImage.ipynb`: collect data using OpenCV. 
>
> `imageToCsv.ipynb`: Transform images to pixels and save in csv file. --> output `train.csv & test.csv` 
>
> `model_simple.ipynb`: A simple CNN model with 1 layer to test data
>
> `model_2.ipynb`: A simple CNN model with 3 layers to test data
> `sing_language_resnet50.ipynb`: ResNet CNN model with 9 layers to test data
> `resnet.model`: trained data from `sing_language_resnet50.ipynb`





