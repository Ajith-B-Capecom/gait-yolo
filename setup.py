"""
Setup Script for OpenGait Project
Creates folder structure and sample person folders
"""

import os
from pathlib import Path


def create_folder_structure():
    """Create the complete folder structure for OpenGait"""
    
    folders = [
        'data/videos/person1',
        'data/videos/person2',
        'data/videos/person3',
        'data/frames',
        'data/detected_persons',
        'data/silhouettes',
        'output',
        'scripts'
    ]
    
    print("\n" + "="*60)
    print("  OpenGait Project Setup")
    print("="*60)
    
    print("\nCreating folder structure...")
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {folder}")
    
    # Create placeholder README in person folders
    person_folders = ['person1', 'person2', 'person3']
    
    for person in person_folders:
        readme_path = f'data/videos/{person}/README.txt'
        if not os.path.exists(readme_path):
            with open(readme_path, 'w') as f:
                f.write(f"Place video files for {person} in this folder.\n")
                f.write(f"Supported formats: .mp4, .avi, .mov, .flv, .mkv\n")
    
    print("\n" + "="*60)
    print("✓ Setup Complete!")
    print("="*60)
    
    print("\nNext Steps:")
    print("1. Activate virtual environment:")
    print("   .\\venv\\Scripts\\Activate.ps1")
    print("\n2. Install requirements:")
    print("   pip install -r requirements.txt")
    print("\n3. Add video files to:")
    print("   data/videos/person1/")
    print("   data/videos/person2/")
    print("   data/videos/person3/")
    print("\n4. Run the pipeline:")
    print("   python main.py")
    print()


if __name__ == "__main__":
    create_folder_structure()
