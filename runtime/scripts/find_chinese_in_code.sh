#!/usr/bin/env bash
# 在代码中查找包含中文的行。实际逻辑在 find_chinese_in_code.py
# 用法:
#   ./scripts/find_chinese_in_code.sh              # 只查 python/ afd/ benchmark/ scripts/ test/ 等
#   ./scripts/find_chinese_in_code.sh --all        # 全仓库（含 3rdparty）
#   ./scripts/find_chinese_in_code.sh --dir python # 只查 python/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/find_chinese_in_code.py" "$@"
