import os

# Define the structure
structure = {
    'projects': {
        'personas': ['__init__.py', 'profile_builder.py', 'modeling_persona.py', 'energy_policy_persona.py'],
        'utils': ['__init__.py', 'data_manager.py'],
        'manager': ['__init__.py', 'persona_manager.py'],
        '': ['main.py', 'requirements.txt']  # Root level files
    }
}

def create_structure(base_path, structure):
    for folder, files in structure.items():
        folder_path = os.path.join(base_path, folder)
        if folder:  # Avoid creating folder for root-level files
            os.makedirs(folder_path, exist_ok=True)
        for file in files:
            file_path = os.path.join(folder_path, file) if folder else os.path.join(base_path, file)
            with open(file_path, 'w') as f:
                pass  # Just creating an empty file

# Specify the base directory
base_dir = 'projects'

# Create the directory structure
create_structure(base_dir, structure)

print("Project structure created successfully!")
