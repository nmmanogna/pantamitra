# PantaMitra
1. pip install -r requirements.txt
2. Download the PlantVillage dataset (Kaggle) and unzip so each class is a folder.
3. python train.py PlantVillage   (prints validation accuracy, saves crop_model.keras + classes.json)
4. streamlit run app.py
Without step 3 the app runs in demo mode (pick a disease, hear the advice).
