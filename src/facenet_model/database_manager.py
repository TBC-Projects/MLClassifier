"""
Database Management Utility
Tools for managing the face recognition database
"""

from face_recognition_pipeline import FaceRecognitionPipeline
from pathlib import Path
import json
import numpy as np

class DatabaseManager:
    def __init__(self, pipeline=None):
        if pipeline is None:
            self.pipeline = FaceRecognitionPipeline()
        else:
            self.pipeline = pipeline
    
    def list_people(self):
        """List all people in the database"""
        print("\n" + "="*50)
        print("PEOPLE IN DATABASE")
        print("="*50)
        
        if len(self.pipeline.face_database) == 0:
            print("Database is empty")
            return
        
        for idx, name in enumerate(self.pipeline.face_database.keys(), 1):
            embedding = self.pipeline.face_database[name]
            print(f"{idx}. {name} (embedding dim: {len(embedding)})")
    
    def remove_person(self, person_name):
        """Remove a person from the database"""
        if person_name in self.pipeline.face_database:
            del self.pipeline.face_database[person_name]
            self.pipeline.save_database()
            print(f"✓ Removed {person_name} from database")
            return True
        else:
            print(f"✗ {person_name} not found in database")
            return False
    
    def clear_database(self):
        """Clear entire database"""
        confirm = input("Are you sure you want to clear the entire database? (yes/no): ")
        if confirm.lower() == 'yes':
            self.pipeline.face_database = {}
            self.pipeline.save_database()
            print("✓ Database cleared")
        else:
            print("Operation cancelled")
    
    def export_database_info(self, output_file="database_info.json"):
        """Export database information to JSON"""
        info = {
            "num_people": len(self.pipeline.face_database),
            "people": list(self.pipeline.face_database.keys()),
            "threshold": self.pipeline.threshold,
            "embedding_dimension": len(next(iter(self.pipeline.face_database.values()))) if self.pipeline.face_database else 0
        }
        
        with open(output_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✓ Database info exported to {output_file}")
    
    def batch_add_from_folder(self, training_data_folder):
        """
        Add all people from a training data folder
        
        Expected structure:
        training_data/
            ├── person1/
            ├── person2/
            └── ...
        """
        training_data_path = Path(training_data_folder)
        
        if not training_data_path.exists():
            print(f"Error: {training_data_folder} does not exist")
            return
        
        success_count = 0
        fail_count = 0
        
        for person_folder in training_data_path.iterdir():
            if person_folder.is_dir():
                person_name = person_folder.name
                print(f"\nProcessing {person_name}...")
                
                if self.pipeline.add_person_to_database(person_name, person_folder):
                    success_count += 1
                else:
                    fail_count += 1
        
        if success_count > 0:
            self.pipeline.save_database()
        
        print(f"\n{'='*50}")
        print(f"Batch processing complete:")
        print(f"  Success: {success_count}")
        print(f"  Failed: {fail_count}")
        print(f"{'='*50}")
    
    def verify_database(self):
        """Verify database integrity"""
        print("\nVerifying database...")
        
        if len(self.pipeline.face_database) == 0:
            print("✗ Database is empty")
            return False
        
        issues = []
        
        for name, embedding in self.pipeline.face_database.items():
            # Check embedding dimension
            if len(embedding) != 512:
                issues.append(f"{name}: incorrect embedding dimension ({len(embedding)})")
            
            # Check for NaN or Inf values
            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                issues.append(f"{name}: contains NaN or Inf values")
        
        if issues:
            print("✗ Issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print(f"✓ Database verified successfully ({len(self.pipeline.face_database)} people)")
            return True
    
    def update_person(self, person_name, new_image_folder):
        """Update embeddings for an existing person"""
        print(f"Updating {person_name}...")
        
        # Remove old entry
        if person_name in self.pipeline.face_database:
            del self.pipeline.face_database[person_name]
        
        # Add new entry
        if self.pipeline.add_person_to_database(person_name, new_image_folder):
            self.pipeline.save_database()
            print(f"✓ Updated {person_name}")
            return True
        else:
            print(f"✗ Failed to update {person_name}")
            return False


def main():
    """Interactive database management"""
    print("="*50)
    print("DATABASE MANAGEMENT UTILITY")
    print("="*50)
    
    manager = DatabaseManager()
    
    while True:
        print("\nOptions:")
        print("1. List all people")
        print("2. Add person from folder")
        print("3. Batch add from training data folder")
        print("4. Remove person")
        print("5. Update person")
        print("6. Clear database")
        print("7. Verify database")
        print("8. Export database info")
        print("9. Exit")
        
        choice = input("\nEnter your choice (1-9): ")
        
        if choice == '1':
            manager.list_people()
        
        elif choice == '2':
            name = input("Enter person name: ")
            folder = input("Enter folder path: ")
            if Path(folder).exists():
                manager.pipeline.add_person_to_database(name, folder)
                manager.pipeline.save_database()
            else:
                print(f"Error: Folder {folder} not found")
        
        elif choice == '3':
            folder = input("Enter training data folder path: ")
            manager.batch_add_from_folder(folder)
        
        elif choice == '4':
            manager.list_people()
            name = input("\nEnter name to remove: ")
            manager.remove_person(name)
        
        elif choice == '5':
            manager.list_people()
            name = input("\nEnter name to update: ")
            folder = input("Enter new image folder path: ")
            if Path(folder).exists():
                manager.update_person(name, folder)
            else:
                print(f"Error: Folder {folder} not found")
        
        elif choice == '6':
            manager.clear_database()
        
        elif choice == '7':
            manager.verify_database()
        
        elif choice == '8':
            filename = input("Enter output filename (default: database_info.json): ")
            if not filename:
                filename = "database_info.json"
            manager.export_database_info(filename)
        
        elif choice == '9':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
