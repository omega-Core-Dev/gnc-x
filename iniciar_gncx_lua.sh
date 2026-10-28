#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "=== MOTOR GEOMÉTRICO GNC-X LLM (LuaJIT FFI Zero-Overhead) ==="
luajit lua/gncx_chat.lua "$@"
