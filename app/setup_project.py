"""
HR Agent Backend - Project Setup Script
Run this to create the correct directory structure and __init__.py files
"""

import os
from pathlib import Path

def create_directory_structure():
    """Create the project directory structure"""
    
    print("🏗️  Creating HR Agent Backend Structure...")
    print()
    
    # Define directory structure
    directories = [
        "app",
        "app/agent",
        "app/services", 
        "app/models"
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}/")
    
    print()
    print("📝 Creating __init__.py files...")
    
    # Create __init__.py files
    init_files = [
        "app/__init__.py",
        "app/agent/__init__.py",
        "app/services/__init__.py",
        "app/models/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✅ Created: {init_file}")
    
    print()
    print("=" * 60)
    print("✅ PROJECT STRUCTURE CREATED!")
    print("=" * 60)
    print()
    print("📋 Next steps:")
    print("1. Copy code files into their respective locations:")
    print("   - config.py → app/config.py")
    print("   - schemas.py → app/models/schemas.py")
    print("   - qdrant_service.py → app/services/qdrant_service.py")
    print("   - rag_service.py → app/services/rag_service.py")
    print("   - decision_engine.py → app/services/decision_engine.py")
    print("   - tools.py → app/agent/tools.py")
    print("   - langchain_agent.py → app/agent/langchain_agent.py")
    print("   - main.py → app/main.py")
    print()
    print("2. Create .env file with your credentials")
    print("3. Install requirements: pip install -r requirements.txt")
    print("4. Run server: uvicorn app.main:app --reload")
    print()
    print("=" * 60)
    
    # Show tree structure
    print("\n📁 Project structure:")
    print("""
hr-agent-backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── langchain_agent.py
│   │   └── tools.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── qdrant_service.py
│   │   ├── rag_service.py
│   │   └── decision_engine.py
│   └── models/
│       ├── __init__.py
│       └── schemas.py
├── requirements.txt
├── .env
└── .gitignore
    """)

if __name__ == "__main__":
    create_directory_structure()