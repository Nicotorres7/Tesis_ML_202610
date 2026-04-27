from __future__ import annotations

import sys
from pathlib import Path

from sat_app.ui import run_app


def _run_with_streamlit() -> None:
    from streamlit.web import cli as stcli

    script_path = str(Path(__file__).resolve())
    sys.argv = ["streamlit", "run", script_path]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is None:
        _run_with_streamlit()
    else:
        run_app()
