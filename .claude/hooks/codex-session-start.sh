#!/usr/bin/env bash
# 세션 시작 시: Codex 리뷰 게이트를 켜고, 로그인 상태를 Claude에게 알린다.
# 실패해도 세션 시작을 막지 않는다.

# Codex CLI가 없으면 (setup 스크립트가 없는 환경) 조용히 종료
command -v codex >/dev/null 2>&1 || exit 0

companion=$(ls -d /root/.claude/plugins/synced/*/codex/scripts/codex-companion.mjs 2>/dev/null | head -1)
if [ -n "$companion" ]; then
  CLAUDE_PLUGIN_DATA="${CLAUDE_PLUGIN_DATA:-/root/.claude/plugins/data/codex-synced}" \
    timeout 20 node "$companion" setup --enable-review-gate --json >/dev/null 2>&1 || true
fi

if ! codex login status 2>&1 | grep -q "Logged in"; then
  echo "Codex(ChatGPT)가 아직 로그인되지 않았습니다. 사용자에게 먼저 알리고, 'codex login --device-auth'를 백그라운드로 실행해 로그인 링크와 일회용 코드를 보여주세요."
fi
exit 0
