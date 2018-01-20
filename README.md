# Two-Stage Convolutional Neural Network (TSCNN)

This is code for paper: `Person Re-identification Using Two-Stage
Convolutional Neural Network` by Yonghui Zhang, Jie Shao, Deqiang
Ouyang and Heng Tao Shen (submitted to ICPR 2018).


## Dependency Required

*   Python2.7+

>   tqdm
>   pandas
>   numpy
>   matplotlib
>   pytorch(cuda)


>   Note that we only ran the code in `Unbuntu16-x64`, but it should be able to run successfully in other Linux distribution systems such as `debian` and `centOS`.

## Running

### 1.Download the datasets.

iLIDS-VID:
[http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html](http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html)

PRID2011:
[https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/)

MARS:
[http://www.liangzheng.com.cn/Project/project_mars.html](http://www.liangzheng.com.cn/Project/project_mars.html)

### 2.Extract epicflow.

Download our another project:
[epicflow-python3](https://github.com/zyoohv/epicflow-python3.git).
More information please see `README.md` file in it.

### 3.Edit the configure file.

Our setting file `base_model/setting.json` looks like:

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

in which the key word `is_running` decides to run this part code or
not. The default setting is used to do experiment in `iLIDS-VID`
dataset. Edit it if needed in other datasets.

### 4.Run `base_model/run.sh`.

## QA

Any question will be replied in `Issues` soon, or email the author:
`zyoohv@163.com`
