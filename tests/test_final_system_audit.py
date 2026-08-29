import json, subprocess, sys
from pathlib import Path

def test_audit_script_exists():
    p=Path('scripts/final_system_audit.py'); assert p.exists()

def test_short_logic_is_explicit():
    s=Path('scripts/final_system_audit.py').read_text(); assert 'SHORT_SIGNAL_WITHOUT_FO_ELIGIBILITY' in s
