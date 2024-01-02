# waper
##waper crack detection
1. make dataset  
```
>-data  
>>  -- bad  
>>>    img1  
>>>     img2  
>>>     ...  
>>  -- good  
>>>     img1  
>>>     img2  
>>>     ...
```
2. create anaconda virtual environment and activate  
```python
conda create -n ENV_NAME python==3.9.*  
conda activate ENV_NAME
```
3. `git clone https://github.com/2jjune/waper.git`
4. `cd waper`
5. `git clone https://github.com/rishigami/Swin-Transformer-TF.git Swin_Transformer_TF`
6. `pip install -r requirements.txt`
7. run `vit.py` or `window_vit.py`  
run example
```python
python vit.py --img_size 224 --batch 8 --validate_rate 0.25 --data_dir "your dataset dir" --model_type "vit_small"
```
<br/>NOTE(img_size = 224(vit) or 384(swin, convnext), model_type = vit_small, vit_large, convnext, swin)

***

In code:
1. Check the GPU is available and ignore warning
2. make imagedatagenerator(split train, validation)
3. set saving directory
4. set callback function
5. build model(vit or swin or convnext)
6. set learning parameters
7. train and save model
8. save metrics graph
9. load saved model and validate, test, measure prediction time, save metrics graphs

