"""
Ledger Automator - Main Streamlit Application
Refactored to use MVC (Model-View-Controller) architecture for better maintainability.

This file serves as the main entry point and delegates to the MVC components.
For the legacy implementation, see the git history.
"""

# Import the new MVC-based application
from app_mvc import main

# Run the application
if __name__ == "__main__":
    main()