import sys
sys.path.append('/home/ttuser/.coworker/wt/c14-bfp8-fastpath/scripts')
try:
    from lever_census import install_child_hook
    install_child_hook()
except Exception:
    pass
