import os
import sys

def run_step(script_path):
    print(f"\n[{'='*50}]")
    print(f"Running: {script_path}")
    print(f"[{'='*50}]\n")
    exit_code = os.system(f"{sys.executable} {script_path}")
    if exit_code != 0:
        print(f"Error executing {script_path}. Exiting.")
        sys.exit(exit_code)

if __name__ == '__main__':
    # Ensure working directory is the project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    scripts = [
        os.path.join("src", "data_preprocessing.py"),
        os.path.join("src", "feature_selection.py"),
        os.path.join("src", "train_model.py"),
        os.path.join("src", "evaluate_model.py")
    ]
    
    for script in scripts:
        if os.path.exists(script):
            run_step(script)
        else:
            print(f"Error: {script} not found.")
            sys.exit(1)
            
    print("\nPipeline execution completed successfully!")
