# Hotfix: restore core/orchestrator.py with _run_llm_text

Fixes:
- AttributeError: 'Orchestrator' object has no attribute '_run_llm_text'

Apply:
```bash
cd ~/projects_python/pc_agent_linux
unzip -o pc_agent_linux_hotfix_orchestrator_runllmtext_v1.zip -d .
python3 -c "import re; print('has _run_llm_text:', bool(re.search(r'def\s+_run_llm_text', open('core/orchestrator.py', encoding='utf-8').read())))"
rm -f core/__pycache__/orchestrator*.pyc 2>/dev/null || true
python3 -m py_compile core/orchestrator.py
python3 app.py
```
