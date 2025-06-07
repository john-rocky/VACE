# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import logging
from . import utils

# Configure logging for import warnings
_logger = logging.getLogger(__name__)

# Check if we should suppress import warnings
SUPPRESS_IMPORT_WARNINGS = os.environ.get('VACE_SUPPRESS_IMPORT_WARNINGS', '0') == '1'

try:
    from . import ltx
    _ltx_available = True
except ImportError as e:
    _ltx_available = False
    if not SUPPRESS_IMPORT_WARNINGS:
        print("ℹ️  LTX-Video not available (optional for WAN-only usage)")
        print("   To install: pip install ltx-video@git+https://github.com/Lightricks/LTX-Video@ltx-video-0.9.1 sentencepiece --no-deps")
        print("   To suppress this message: export VACE_SUPPRESS_IMPORT_WARNINGS=1")

try:
    from . import wan
    _wan_available = True
except ImportError as e:
    _wan_available = False
    if not SUPPRESS_IMPORT_WARNINGS:
        print("⚠️  WAN-Video not available - required for WAN model usage")
        print("   To install: pip install wan@git+https://github.com/Wan-Video/Wan2.1")

# Export availability flags
__all__ = ['utils', '_ltx_available', '_wan_available']

if _ltx_available:
    __all__.append('ltx')
if _wan_available:
    __all__.append('wan')
