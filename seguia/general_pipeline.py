

import subprocess

command_list = [
    # This gets drought, meteorological and municipal data
    'python src/data/downloader.py',
    # Now we need to process data to compute features.
    'python src/features/process_data.py',
    # Compute features.
    'python src/features/create_target_and_features.py',
    # Model: This splits data and creates a model.
    'python src/models/create_model.py',
    # Evaluate
    # 'python src/models/evaluate_model.py',
    # Demo:
    # 'python src/demo/app.py'
]

for command in command_list:
    subprocess.run(command.split(' '))
