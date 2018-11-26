I have done this project as an extension to OCR text recognition(bank check's) done by [Adrian](https://www.pyimagesearch.com/category/optical-character-recognition-ocr/)

I have upgraded the project by adding deep learning to it, instead of conventional method to predict what character it is on the check(which is done by comparing), I have trained a CNN(Convolution neural network ) to predict the respective text with font( MICR E-13B )

Implementation-:

1) Clone the repo

2) Install the necessary packages:
```
pip install keras
pip install skimage
pip install cv2
pip install imutils

```

3) navigate to OCR directory
```
cd OCR
```

4) run the model.py file

```
python model.py
```
