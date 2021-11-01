# FDRNet
Code for our ICCV 2021 paper "[Mitigating Intensity Bias in Shadow Detection via Feature Decomposition and Reweighting](https://openaccess.thecvf.com/content/ICCV2021/html/Zhu_Mitigating_Intensity_Bias_in_Shadow_Detection_via_Feature_Decomposition_and_ICCV_2021_paper.html)"


## How to Use

### create conda environment

```
conda env create -f env.yaml
```

To use CRF refinement, you will need to mannually install [pydensecrf](https://github.com/Andrew-Qibin/dss_crf).

**WARNING**: To reproduce the results reported in our paper, please make sure major pacakges (pytorch, opencv, etc.) are with the same version speficified in `env.yaml`.

### run inference

* download the checkpoint from [here]().
* specify data_root, and run `python test.py`. 
* run `python crf_refine.py`.
* check the results w/ and w/o CRF refinement in `test/raw` and `test/crf` respectively

## Results



