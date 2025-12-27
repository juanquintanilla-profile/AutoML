"""
AutoML Agent - Package entry point
Allows running: python -m automl_agent
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from .main import main

if __name__ == "__main__":
    main()
