"""
blender_navigator.py
Blender Action & Workflow Planner for AI Agents and Users.

Translates high-level natural language goals into step-by-step procedural action plans:
"Go there, switch to that mode, click on that menu/button, press that shortcut, adjust settings."
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import warnings
import io
import contextlib

# Suppress HuggingFace hub warnings and parallel tokenizers warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

WORKSPACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.db import BlenderGraphDB

# Global embedding model cache
_EMBEDDING_MODEL = None


def get_embedding_model():
    """Lazily load SentenceTransformer model for fast query embeddings without polluting stdout/stderr."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            null_buf = io.StringIO()
            with contextlib.redirect_stdout(null_buf), contextlib.redirect_stderr(null_buf):
                from sentence_transformers import SentenceTransformer
                _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMBEDDING_MODEL = False
    return _EMBEDDING_MODEL if _EMBEDDING_MODEL is not False else None


def format_procedural_steps(target_node: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
    """
    Synthesizes exact procedural instructions:
    "Go there, switch mode, click on that, press shortcut, adjust settings."
    """
    steps = []

    name = target_node.get("name", "Target Tool")
    category = target_node.get("category", "")
    mode = target_node.get("mode")
    menu_path = target_node.get("menu_path")
    shortcut = target_node.get("shortcut")
    summary = target_node.get("summary", "")

    # 1. Determine Primary Editor
    editors = context.get("editors", [])
    editor_name = editors[0]["name"] if editors else ("3D Viewport" if category in ("modeling", "sculpt_paint", "interface") else "Properties Editor")

    steps.append(f"Go to the **{editor_name}**.")

    # 2. Mode requirement
    if mode:
        if "Edit" in mode:
            steps.append("Select your target mesh object in the 3D Viewport, then switch to **Edit Mode** (press `Tab`).")
            if category == "modeling" and any(w in name.lower() for w in ["bevel", "extrude", "subdivide", "loop cut", "inset"]):
                steps.append("Select the vertices, edges, or faces you want to operate on (or press `A` to select all).")
        elif "Sculpt" in mode:
            steps.append("Select the mesh object, then switch to **Sculpt Mode** (press `Ctrl + Tab` and choose Sculpt Mode).")
        elif "Pose" in mode:
            steps.append("Select the Armature object, then switch to **Pose Mode** (press `Ctrl + Tab`).")
        elif "Object" in mode:
            steps.append("Ensure you are in **Object Mode** (press `Tab` if currently in Edit Mode).")
    elif category == "modeling":
        steps.append("Ensure your object is selected in the 3D Viewport.")

    # 3. Primary Action (Menu / Panel / Shortcut)
    if menu_path:
        # Check if it's in Properties
        if "Render" in menu_path or "Film" in menu_path or "Properties" in editor_name:
            # Parse panel hierarchy (e.g. Render -> Film -> Transparent)
            parts = [p.strip() for p in menu_path.split("->")]
            if len(parts) >= 2:
                tab = parts[0]
                panel = parts[1]
                steps.append(f"In the Properties Editor sidebar, click on the **{tab}** tab.")
                steps.append(f"Expand the **{panel}** panel.")
                if len(parts) >= 3:
                    target_control = parts[2]
                    steps.append(f"Enable or configure the **{target_control}** option.")
                else:
                    steps.append(f"Configure the **{panel}** settings.")
            else:
                steps.append(f"Navigate to **{menu_path}** in the Properties panel.")
        elif shortcut:
            steps.append(f"Click on menu: **{menu_path}** (or directly press shortcut `{shortcut}`).")
        else:
            steps.append(f"Click on menu: **{menu_path}**.")
    elif shortcut:
        steps.append(f"Press shortcut `{shortcut}` to activate **{name}**.")
    elif "Properties" in editor_name or category in ("physics", "render"):
        tab_hint = "Modifier Properties" if "modifier" in name.lower() else ("Physics Properties" if category == "physics" else "Render Properties")
        steps.append(f"In the Properties Editor sidebar, click on the **{tab_hint}** tab.")
        steps.append(f"Click on **{name}** to enable or configure it.")
    else:
        steps.append(f"Trigger **{name}** from the toolbar or press `F3` and search for \"{name}\".")

    # 4. Operator Settings / Last Operation Panel
    if category == "modeling" or "modifier" in name.lower():
        steps.append("In the **'Adjust Last Operation'** popup panel (bottom-left corner of the viewport, or press `F9`), adjust settings as desired.")
    elif category == "physics":
        steps.append("In the Physics settings panel, adjust simulation parameters (e.g. Quality Steps, Stiffness, Damping) as needed.")

    return steps


def plan_action(query: str, db_path: Optional[str] = None, top_k: int = 1) -> Dict[str, Any]:
    """
    Given a natural language request, generates a procedural plan.
    Returns a dictionary with query, target feature, steps, shortcuts, and python code.
    """
    db = BlenderGraphDB(db_path)
    model = get_embedding_model()

    query_emb = None
    if model:
        query_emb = model.encode(query, normalize_embeddings=True)

    # Hybrid Search (Dense vector + FTS5 BM25)
    matches = db.hybrid_search(query, query_embedding=query_emb, top_k=max(top_k, 5))

    if not matches:
        return {
            "query": query,
            "status": "not_found",
            "message": "No matching Blender documentation or tools found for this query."
        }

    target = matches[0]
    context = db.get_procedural_context(target["id"])
    steps = format_procedural_steps(target, context)

    # Python code for automation
    python_snippet = target.get("python_op")
    if not python_snippet:
        if "modifier" in target["name"].lower():
            mod_type = target["name"].upper().replace(" ", "_")
            python_snippet = f"import bpy\nbpy.ops.object.modifier_add(type='{mod_type}')"
        elif target.get("shortcut") == "Shift-A":
            python_snippet = "import bpy\nbpy.ops.mesh.primitive_cube_add()"
        elif "transparent" in target["name"].lower():
            python_snippet = "import bpy\nbpy.context.scene.render.film_transparent = True"
        elif "cloth" in target["name"].lower():
            python_snippet = "import bpy\nbpy.ops.object.modifier_add(type='CLOTH')"

    plan = {
        "query": query,
        "target_feature": target["name"],
        "category": target.get("category"),
        "mode": target.get("mode"),
        "menu_path": target.get("menu_path"),
        "shortcut": target.get("shortcut"),
        "summary": target.get("summary"),
        "steps": steps,
        "python_code": python_snippet,
        "alternatives": [
            {"name": m["name"], "category": m.get("category"), "score": m.get("score")}
            for m in matches[1:top_k + 2]
        ]
    }

    return plan


def print_cli_plan(plan: Dict[str, Any]):
    """Pretty prints the action plan to the terminal."""
    print("=" * 65)
    print(f" BLENDER WORKFLOW PLANNER")
    print("=" * 65)
    print(f"Goal: \"{plan['query']}\"\n")
    print(f"Target Feature : {plan['target_feature']}")
    if plan.get("category"):
        print(f"Category       : {plan['category'].title()}")
    if plan.get("mode"):
        print(f"Required Mode  : {plan['mode']}")
    if plan.get("shortcut"):
        print(f"Key Shortcut   : {plan['shortcut']}")
    if plan.get("menu_path"):
        print(f"Menu Path      : {plan['menu_path']}")
    print("-" * 65)
    print("PROCEDURAL ACTION STEPS:")
    for i, step in enumerate(plan["steps"], 1):
        clean_step = step.replace("**", "").replace("`", "")
        print(f"  {i}. {clean_step}")

    if plan.get("python_code"):
        print("-" * 65)
        print("PYTHON API AUTOMATION (bpy.ops / bpy.context):")
        for line in plan["python_code"].splitlines():
            print(f"  {line}")

    if plan.get("alternatives"):
        print("-" * 65)
        print("Related Features / Alternatives:")
        for alt in plan["alternatives"][:3]:
            print(f"  - {alt['name']} ({alt.get('category', 'tool')})")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Blender Workflow & Action Navigator for AI Agents.")
    parser.add_argument("query", type=str, help="What do you want to do in Blender? (e.g. 'bevel edges', 'cloth simulation', 'cycles transparent background')")
    parser.add_argument("--json", action="store_true", help="Output result as structured JSON")
    parser.add_argument("--top", type=int, default=1, help="Number of alternatives to inspect")
    args = parser.parse_args()

    plan = plan_action(args.query, top_k=args.top)

    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print_cli_plan(plan)


if __name__ == "__main__":
    main()
