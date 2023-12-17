# Web demo

## How to run web demo
1. Download required packages
```
pip install gdown Flask
```

2. Download demo files
```
gdown --folder --remaining-ok https://drive.google.com/drive/folders/1EToI2PWMHeoPsTXRbnbq0Z77SdBawRH2 -O ./static
```

3. Run `python3 app.py` and you can access the website with the url provided at the console. If the port 5000 is already used in your machine, you can use the following command to choose a custom port.
```
python3 app.py --port $PORT
```

4. The checked result will be saved at `static/result.csv`