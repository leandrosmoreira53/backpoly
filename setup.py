"""
setup.py — Compila o módulo Cython sim_core com OpenMP + flags SIMD.

Uso:
    python setup.py build_ext --inplace

Flags:
    -O3             Otimização máxima
    -march=native   SIMD automático (AVX2/AVX-512 se disponível)
    -ffast-math     Matemática rápida (assume aritmética associativa)
    -fopenmp        OpenMP (prange paralelo)
"""
from __future__ import annotations

import platform
import sys
from setuptools import setup, Extension

import numpy as np

try:
    from Cython.Build import cythonize
except ImportError:
    print("Cython não encontrado. Instale: pip install cython", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Flags de compilação
# ---------------------------------------------------------------------------
extra_compile: list[str] = ["-O3", "-ffast-math"]
extra_link: list[str] = []

_system = platform.system()

if _system == "Darwin":
    # macOS: tentar llvm-openmp (brew install libomp)
    try:
        import subprocess
        result = subprocess.run(
            ["brew", "--prefix", "libomp"], capture_output=True, text=True
        )
        if result.returncode == 0:
            prefix = result.stdout.strip()
            extra_compile += [f"-I{prefix}/include", "-fopenmp=libomp", "-march=native"]
            extra_link    += [f"-L{prefix}/lib", "-lomp"]
        else:
            # sem OpenMP no macOS → -march=native apenas
            extra_compile.append("-march=native")
    except Exception:
        extra_compile.append("-march=native")
else:
    # Linux (e Windows com MinGW)
    extra_compile += ["-march=native", "-fopenmp"]
    extra_link    += ["-fopenmp"]

# ---------------------------------------------------------------------------
# Extensão Cython
# ---------------------------------------------------------------------------
ext = Extension(
    name="src.cy.sim_core",
    sources=["src/cy/sim_core.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=extra_compile,
    extra_link_args=extra_link,
    language="c",
)

# ---------------------------------------------------------------------------
# Compiler directives — todas as otimizações de segurança desligadas
# (arrays internos são controlados, não há acesso fora dos limites)
# ---------------------------------------------------------------------------
compiler_directives: dict = {
    "language_level":   "3",
    "boundscheck":      False,
    "wraparound":       False,
    "cdivision":        True,
    "nonecheck":        False,
    "initializedcheck": False,
    "embedsignature":   True,   # docs úteis no help()
}

setup(
    name="polymarket_bt",
    version="1.0.0",
    ext_modules=cythonize(
        [ext],
        compiler_directives=compiler_directives,
        annotate=True,          # gera sim_core.html para profiling Cython
    ),
)
