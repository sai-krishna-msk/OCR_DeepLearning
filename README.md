- This project is an extension to OCR text recognition(bank checks) done by Adrian you can find it [here]((https://www.pyimagesearch.com/category/optical-character-recognition-ocr/), Where he uses conventional approach and suggests that ML or DL could give better results so I thought to give it a shot, 



- I have improved the accuracy of the prediction of text by training a deep learning model(CNN) on a dataset of bank check's with font style(MICR E-138)



### Implementation



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
