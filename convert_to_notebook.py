import json
import nbformat as nbf

# Read the Python file
with open('motogp_model_improvement.py', 'r') as f:
    python_code = f.read()

# Split the code into sections based on comments
sections = python_code.split('#')

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add cells
cells = []

# Add title cell
cells.append(nbf.v4.new_markdown_cell("# MotoGP Lap Time Prediction - Model Improvement\n\nThis notebook focuses on improving the model performance through feature importance analysis and hyperparameter tuning."))

# Add import cell
import_section = sections[0].strip()
cells.append(nbf.v4.new_code_cell(import_section))

# Add data loading section
data_loading = sections[1].strip()
cells.append(nbf.v4.new_markdown_cell("## Load and Prepare Data"))
cells.append(nbf.v4.new_code_cell(data_loading))

# Add preprocessing section
preprocessing = sections[2].strip()
cells.append(nbf.v4.new_markdown_cell("## Data Preprocessing"))
cells.append(nbf.v4.new_code_cell(preprocessing))

# Add feature importance section
feature_importance = sections[3].strip()
cells.append(nbf.v4.new_markdown_cell("## Feature Importance Analysis"))
cells.append(nbf.v4.new_code_cell(feature_importance))

# Add model training section
model_training = sections[4].strip()
cells.append(nbf.v4.new_markdown_cell("## Model Training and Hyperparameter Tuning"))
cells.append(nbf.v4.new_code_cell(model_training))

# Add model evaluation section
model_evaluation = sections[5].strip()
cells.append(nbf.v4.new_markdown_cell("## Model Evaluation"))
cells.append(nbf.v4.new_code_cell(model_evaluation))

# Add test predictions section
test_predictions = sections[6].strip()
cells.append(nbf.v4.new_markdown_cell("## Generate Predictions for Test Data"))
cells.append(nbf.v4.new_code_cell(test_predictions))

# Add cells to notebook
nb['cells'] = cells

# Add metadata
nb['metadata'] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.8.0"
    }
}

# Write the notebook to file
with open('motogp_model_improvement.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Conversion completed successfully!") 