#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "[GNC-X] Compilando kernel C++20 para Linux nativo (x86_64)..."

mkdir -p bin

if command -v clang++ &> /dev/null; then
    COMPILER="clang++"
elif command -v g++ &> /dev/null; then
    COMPILER="g++"
else
    echo "ERRO: Nenhum compilador C++20 (g++ ou clang++) encontrado."
    echo "Instale no Fedora com: sudo dnf install -y gcc-c++"
    exit 1
fi

echo "[GNC-X] Usando: $COMPILER"
$COMPILER -O3 -std=c++20 -march=native -shared -fPIC -o bin/gncx_llm_core.so csrc/gncx_llm_kernels.cpp -lpthread

echo "[GNC-X] Compilação concluída com sucesso: bin/gncx_llm_core.so"
