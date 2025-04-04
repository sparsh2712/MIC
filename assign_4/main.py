"""
Main script to run all questions
"""
import os
import argparse
from q1 import q1
from q2 import q2
from q3 import q3
from q4 import q4
from utils.io_utils import create_directory, ensure_data_extracted

def main():
    parser = argparse.ArgumentParser(description='Run assignment 4 solutions')
    parser.add_argument('--questions', type=str, default='all',
                        help='Comma-separated list of questions to run (e.g., "1,3,4") or "all"')
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    data_dir = 'assign4_data'
    create_directory(data_dir)
    
    # Ensure data is extracted
    ensure_data_extracted(data_dir)
    
    # Determine which questions to run
    if args.questions.lower() == 'all':
        questions = [1, 2, 3, 4]
    else:
        questions = [int(q) for q in args.questions.split(',')]
    
    # Run selected questions
    if 1 in questions:
        print("\nRunning Question 1")
        q1()
    
    if 2 in questions:
        print("\nRunning Question 2")
        q2()
    
    if 3 in questions:
        print("\nRunning Question 3")
        q3()
    
    if 4 in questions:
        print("\nRunning Question 4")
        q4()
    
    print("\nAll selected questions completed.")

if __name__ == "__main__":
    main()