# Two-Stage Convolutional Neural Network (TSCNN)

This is code for paper:  
`Person Re-identification Using Two-stage Convolutional Neural Network`


## Dependency Required

*   Python2.7+

>   tqdm  
>   pandas  
>   numpy  
>   matplotlib  
>   pytorch(cuda)


>   Note that we only ran the code in `Unbuntu16-x64`, but it should can be ran successfully in other Linux distribution systems such as `debian` and `centOS`.

## Runing

1. Download the dataset.

iLIDS-VID:[http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html](http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html)

PRID2011:[https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/)

MARS:[http://www.liangzheng.com.cn/Project/project_mars.html](http://www.liangzheng.com.cn/Project/project_mars.html)

2. Extract epicflow.

Download our another project: [epicflow-python3](https://github.com/zyoohv/epicflow-python3.git). More information please reference `README.md` file in it.


3. edit the configure file `base_model/setting.json`

Our setting file looks like:

```
{
    "generate_dic": {
        ...
        "is_running": true
    },

    "model": {
        ...
        "is_running": false,
        ...
    },

    "full_model": {
        ...
        "is_running": false,
        ...
    }
}
```

In which the key word `is_running` decide if run this part code. The default setting is used to do experiment in `iLIDS-VID` dataset, edit it if needed in others datasets.

4. run `base_model/run.sh`

## QA

Any question will be relied in `Issues` soon, or email the author: `zyoohv@163.com`