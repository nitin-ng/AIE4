import os

model_path = 'finetuned_arctic'
if os.path.exists(model_path):
    print(f"Model directory exists. Contents:")
    for file in os.listdir(model_path):
        print(f" - {file}")
else:
    print(f"Model directory {model_path} does not exist.")
