#!/usr/bin/env bash
# scripts/package_for_vultr.sh - Bundles the Thunder project for deployment
set -euo pipefail

PACKAGE_NAME="thunder_deploy.tar.gz"
echo "📦 Packaging Thunder for Vultr deployment..."

# Create a temporary exclusion list
CAT_EXCLUDES=$(mktemp)
cat <<EOF > "$CAT_EXCLUDES"
.git
.venv
venv
__pycache__
*.pyc
runs
wandb
*.log
*.tar.gz
.gemini
.env
EOF

# Package the repository
tar -czf "$PACKAGE_NAME" -X "$CAT_EXCLUDES" .

echo "✅ Done! Created $PACKAGE_NAME"
echo "🚀 Next step: Upload this file to your Vultr instances."
rm "$CAT_EXCLUDES"
