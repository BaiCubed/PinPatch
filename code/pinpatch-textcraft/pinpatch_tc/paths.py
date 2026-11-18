"""All paths are relative to the project root (the folder that contains pinpatch_tc/)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD = os.path.join(ROOT, 'third_party')
TEXTCRAFT_PARENT = THIRD                      # contains textcraft/ (ADaPT env code + recipes/)
TEXTCRAFT_DIR = os.path.join(THIRD, 'textcraft') + '/'
ADAPT_RESULTS = os.path.join(THIRD, 'ADaPT', 'results', 'ADaPT_maxd3_textcraft_gpt-3.5-turbo-instruct.json')
WORK = os.path.join(ROOT, os.environ.get('METAACT_WORK', 'workdir')) + '/'   # models, logs and results written by run.py
DATA = os.path.join(ROOT, 'data') + '/'
V154_CODE = os.environ.get('V154_CODE', '')  # optional: closeout package frozen-v154 code dir
def setup_sys_path():
    os.environ.setdefault('TEXTCRAFT_RECIPE_ORDER', os.path.join(ROOT, 'pinpatch_tc', 'recipe_order.json'))
    for p in (ROOT, TEXTCRAFT_PARENT):
        if p not in sys.path: sys.path.insert(0, p)
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        sys.path.insert(0, os.path.join(ROOT, 'shim'))
