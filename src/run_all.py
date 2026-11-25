"""Run the whole pipeline in order (about 5-10 minutes; modelling is the slow step)."""
import subprocess, sys
from pathlib import Path

here = Path(__file__).parent
for script in ["01_clean_data.py", "02_feature_engineering.py", "04_eda_charts.py",
               "03_modeling.py", "05_model_charts.py", "06_report_card.py"]:
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([sys.executable, str(here / script)], check=True, cwd=here)
print("\nDone. See charts/, outputs/ and data/clean/.")
