# Install Dependencies

Install the necessary dependencies for simile component extraction.
```
cd HAUSER/dependency
bash install.sh
python test_syntax.py
```
If test_syntax.py runs successfully, it proves that the dependency installation is successful.

Cache the necessary pre-trained models
```
cd HAUSER/dependency
python test_models.py
```

Download the Million-scale Simile Knowledge Base [MAPS-KB](https://arxiv.org/abs/2212.05254) from [Google Drive](https://drive.google.com/file/d/1d-Xn9OygjxhMoGPoMXSv48-etnOHjkve/view).

Put the file **MAPS-KB.csv**  into the folder [HAUSER/dependency].