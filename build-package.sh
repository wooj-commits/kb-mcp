#!/usr/bin/env bash
# Builds an install-ready kbmcp package with tokens baked into values.
# Never prints token values.
set -euo pipefail
umask 077
ROOT=/workspace/kb-mcp
TPL=$ROOT/chart/kbmcp
KEYF=/workspace/secrets/kb_mcp_tokens
B=/workspace/secrets/build-kbmcp
OUT=$ROOT
H=${HELM:-/workspace/tools/helm}

TOKENS=$(paste -sd, "$KEYF")
redact() {
  # redact every token value (after the colon)
  python3 - "$KEYF" <<'PY'
import sys
tokens=[line.strip().split(":",1)[1] for line in open(sys.argv[1]) if ":" in line]
text=sys.stdin.read()
for t in tokens:
    text=text.replace(t, "<REDACTED>")
sys.stdout.write(text)
PY
}

rm -rf "$B"; mkdir -p "$B"
cp -a "$TPL" "$B/"
python3 - "$KEYF" "$B/kbmcp/values.yaml" <<'PY'
import sys
pairs=[]
for line in open(sys.argv[1]):
    line=line.strip()
    if not line or ":" not in line: continue
    name, tok = line.split(":",1)
    assert len(tok) >= 32, name
    pairs.append(f"{name}:{tok}")
joined=",".join(pairs)
p=sys.argv[2]
s=open(p).read()
old='  MCP_TOKENS: ""'
assert old in s, "placeholder missing"
s=s.replace(old, f'  MCP_TOKENS: "{joined}"')
open(p,"w").write(s)
print("tokens injected into values.yaml (names only):", ",".join(n.split(":")[0] for n in pairs))
PY

cd "$B"
VER=$(python3 -c "import yaml;print(yaml.safe_load(open('kbmcp/Chart.yaml'))['version'])")
MVER=$(python3 -c "import yaml;print(yaml.safe_load(open('kbmcp/OlaresManifest.yaml'))['metadata']['version'])")
[ "$VER" = "$MVER" ] || { echo "version mismatch"; exit 1; }
echo "chart version $VER"
if [ -x "$H" ]; then
  $H lint kbmcp 2>&1 | redact | tail -5
  $H template kbmcp ./kbmcp --namespace kbmcp-woojinc1205 \
    --set userspace.appData=/tmp/appdata --set userspace.userData=/tmp/home \
    --set env.ROUTER_BASE_URL=http://router.example/v1 \
    --set env.CHAT_MODEL=demo --set env.EMBED_MODEL=embed \
    > rendered.yaml
  echo "helm template ok; objects:" $(grep -c '^kind:' rendered.yaml || true)
  $H package ./kbmcp -d "$B/pkg" >/dev/null
  # Do NOT copy the token-baked tgz to a world-readable place as the default deliverable;
  # leave it under /workspace/secrets. The key-less chart stays in chart/kbmcp.
  chmod 600 "$B/pkg/kbmcp-$VER.tgz"
  echo "packaged (secrets dir): $B/pkg/kbmcp-$VER.tgz"
else
  echo "helm not found at $H; skipped lint/package"
fi
