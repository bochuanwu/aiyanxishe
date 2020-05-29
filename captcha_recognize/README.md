## Train
```angular2html
   [run] python train.py --cfg lib/config/360CC_config.yaml
or [run] python train.py --cfg lib/config/OWN_config.yaml
```
```
#### loss curve

```angular2html
   [run] cd output/360CC/crnn/xxxx-xx-xx-xx-xx/
   [run] tensorboard --logdir log
```

## Demo
```angular2html
   [run] python demo.py 
```
## References
- https://github.com/meijieru/crnn.pytorch
- https://github.com/HRNet



