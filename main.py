import os
import sys
import uvicorn

# Get the absolute path of the current file (main.py)
project_root = os.path.dirname(os.path.abspath(__file__))

# Insert the project root at the beginning of sys.path if it's not already there.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ == "__main__":
    # Start the FastAPI server defined in api/server.py
    uvicorn.run("modules.api.server:app", host="0.0.0.0", port=8000, reload=False)
