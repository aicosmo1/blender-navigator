# Blender Navigator - Portable AI Agent & Workflow Engine

An offline Knowledge Graph, Vector Index, and Procedural Action Planner for **Blender 5.2.2 LTS**.

When you ask how to do something in Blender (e.g., *"I want to bevel the edges of a cube"* or *"make cycles render background transparent"*), this engine queries the local graph database (`./db/blender_graph.db`) and generates exact, step-by-step UI actions:
1. **Where to go** (Editor & Mode)
2. **What to click** (Menu path, Panel, Button)
3. **What shortcut to press** (Keymap / hotkey)
4. **What settings to tweak** (Operator properties & options)
5. **Python Automation code** (`bpy.ops` / `bpy.context`) for AI agents.

---

## Package Contents

- **`blender_navigator.py`**: Main CLI and Python module for agents and users.
- **`db/blender_graph.db`**: Pre-indexed database (39.6 MB) containing:
  - 8,461 nodes (Editors, Modes, Operators, Panels, Sub-settings, Shortcuts)
  - 16,970 semantic graph edges
  - Full-Text Search (FTS5 BM25) index
  - 8,461 dense vector embeddings (`all-MiniLM-L6-v2`)
- **`docs/blender_manual/`**: The complete 2,358 official Blender 5.2.2 LTS manual files (RST source, 5.65 MB).
- **`src/db.py`**: SQLite graph driver and hybrid search engine.
- **`requirements.txt`**: Minimal dependencies list.
- **`setup.bat`**: 1-click environment installer for Windows.
- **`run.bat`**: 1-click query runner.

---

## Quickstart

### Method 1: Using the 1-Click Scripts (Windows)

1. Double click **`setup.bat`** (or run `setup.bat` in Command Prompt) to create `.venv` and install dependencies.
2. Run any query with **`run.bat`**:
   ```cmd
   run.bat "bevel the edges of a cube"
   run.bat "make cycles render background transparent"
   run.bat "add cloth simulation to a plane" --json
   ```

### Method 2: Manual Setup (Windows / macOS / Linux)

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Run queries!
python blender_navigator.py "bevel edges of a cube"
```

---

## Python API Usage (for AI Agents)

You can import `blender_navigator` directly into your agent scripts:

```python
from blender_navigator import plan_action

# Ask for a procedural action plan
plan = plan_action("make cycles render background transparent")

print("Target Feature:", plan["target_feature"])
print("Required Mode:", plan["mode"])
print("Menu Path:", plan["menu_path"])

print("\nSteps:")
for i, step in enumerate(plan["steps"], 1):
    print(f"  {i}. {step}")

# Automate in Blender via Python API:
if plan.get("python_code"):
    print("\nPython Code:")
    print(plan["python_code"])
```

### JSON Mode for Agent Tooling

To get machine-readable output:
```bash
python blender_navigator.py "add cloth simulation to a plane" --json
```

Returns:
```json
{
  "query": "add cloth simulation to a plane",
  "target_feature": "Cloth Dynamics",
  "category": "modeling",
  "steps": [
    "Go to the **3D Viewport**.",
    "Ensure your object is selected in the 3D Viewport.",
    "Trigger **Cloth Dynamics** from the toolbar or press `F3` and search for \"Cloth Dynamics\".",
    "In the **'Adjust Last Operation'** popup panel (bottom-left corner of the viewport, or press `F9`), adjust settings as desired."
  ],
  "python_code": "import bpy\nbpy.ops.object.modifier_add(type='CLOTH')"
}
```
