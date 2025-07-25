import ctypes
import os

lib = ctypes.CDLL('./libclm_lsm.so')
print("CLM shared library loaded successfully.")
