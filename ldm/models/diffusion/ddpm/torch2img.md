```python
def torch2img(input):
    input_ = input[0]
    input_ = input_.permute(1,2,0)
    input_ = input_.data.cpu().numpy()
    input_ = (input_ + 1.0) / 2
    cv2.imwrite('./test.png', input_[:,:,::-1]*255.0)
```

