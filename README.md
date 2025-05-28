https://www.kaggle.com/datasets/muratkokludataset/rice-msc-dataset/data

Use the above, download the dataset as a zip.

From the zip, take only the "Rice_MSC_Dataset.xlsx" file and put it in the "/dataset" folder.

Do the folder in the root folder:
1. Create a .venv through "python -m venv .venv". This was done through Python 3.12.2.
2. Enter in the terminal: "./.venv/Scripts/activate"
3. "pip install -r ./requirements.txt"
4. "py ./train_and_save_models.py"
5. "uvicorn src.main:app --reload"

The server should now work.

There are 3 APIs:
1. /predict -> gives predictions based on the 106 features, given that the model name ("knn", "svm", "nb" is provided)
2. /projection -> Dimensionally reduces the dataset into 2D, provides the final result
3. /metrics/{svm/knn/nb} -> Provides accuracy metrics (conf matrix) for any of SVM, KNN, or NB models
