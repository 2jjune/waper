# waper
waper crack detection
1. make dataset
-data
  -- bad
     img1
     img2
     ...
  -- good
     img1
     img2
     ...
2. git clone https://github.com/rishigami/Swin-Transformer-TF.git Swin_Transformer_TF     
3. run "vit.py" or "window_vit.py"

In code:
1. Check the GPU is available and ignore warning
2. set parameter(img size, dir, etc..)
3. make imagedatagenerator(split train, validation)
4. build model(vit or swin or convnext)
5. make plot metrics function
6. set save directory
7. set callback function
8. load model that made at 4. and set learning parameter
9. train and save
10. load saved model and validate, test, measure prediction time, save metrics graphs

