#!/usr/bin/env bash
# 把 private 分支的白名单子集导出为公开快照，推送到公开仓库（默认 GitHub origin/main）。
# Export the whitelisted subset of the private branch to the public repository.
#
#   bash scripts/publish_public.sh            # 生成提交并推送
#   bash scripts/publish_public.sh --dry-run  # 只显示将发生的变化
#
# 可通过环境变量覆盖：SRC_BRANCH（默认 private）、DST_BRANCH（默认 main）、REMOTE（默认 origin）
set -euo pipefail

SRC_BRANCH=${SRC_BRANCH:-private}
DST_BRANCH=${DST_BRANCH:-main}
REMOTE=${REMOTE:-origin}

# 允许出现在公开仓库的路径（其余一律排除）
PUBLIC_PATHS=(
  .gitignore
  README.md
  datasets/preprocess.ipynb
  icl
  retrieval
  scripts
)

# 只在私有侧存在、不能随 scripts/ 一起发布的路径
PRIVATE_PATHS=(
  scripts/public_overrides
  scripts/local_env.sh
)

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

cd "$(git rev-parse --show-toplevel)"

git rev-parse --verify -q "$SRC_BRANCH" > /dev/null || {
  echo "source branch '$SRC_BRANCH' not found" >&2
  exit 1
}

tmp_index=$(mktemp -u "${TMPDIR:-/tmp}/publish-index.XXXXXX")
trap 'rm -f "$tmp_index"' EXIT
export GIT_INDEX_FILE=$tmp_index

git read-tree --empty

# 1) 只加入白名单路径，并排除私有子路径
exclude_re="$(printf '\t'"(%s)($|/)" "$(printf '%s\n' "${PRIVATE_PATHS[@]}" | sed 's/[].[/*^$\\]/\\&/g' | paste -sd '|' -)")"
git ls-tree -r "$SRC_BRANCH" -- "${PUBLIC_PATHS[@]}" \
  | grep -Ev "$exclude_re" \
  | git update-index --index-info

# 2) 用 public_overrides/ 内的文件覆盖同名路径（脱敏 / 裁剪后的公开版本）
while IFS=$'\t' read -r meta path; do
  rel=${path#scripts/public_overrides/}
  mode=${meta%% *}
  sha=${meta##* }
  git update-index --add --cacheinfo "$mode,$sha,$rel"
done < <(git ls-tree -r "$SRC_BRANCH" -- scripts/public_overrides)

tree=$(git write-tree)
src_sha=$(git rev-parse --short "$SRC_BRANCH")

parent=$(git rev-parse --verify -q "refs/remotes/$REMOTE/$DST_BRANCH" || true)
if [ -n "$parent" ] && [ "$tree" = "$(git rev-parse "$parent^{tree}")" ]; then
  echo "没有变化，$REMOTE/$DST_BRANCH 已是最新（$(git rev-parse --short "$parent")）"
  exit 0
fi

echo "公开快照：$SRC_BRANCH@$src_sha -> $REMOTE/$DST_BRANCH"
if [ -n "$parent" ]; then
  git diff --stat "$parent" "$tree"
  echo
fi

if [ "$DRY_RUN" = 1 ]; then
  echo "将发布以下文件："
  git ls-files | sed 's/^/  /'
  echo "[dry-run] 未提交、未推送"
  exit 0
fi

message="public snapshot of $SRC_BRANCH@$src_sha ($(date +%Y-%m-%d))"
if [ -n "$parent" ]; then
  new_commit=$(git commit-tree "$tree" -p "$parent" -m "$message")
  git update-ref "refs/heads/$DST_BRANCH" "$new_commit" "$parent"
else
  new_commit=$(git commit-tree "$tree" -m "$message")
  git update-ref "refs/heads/$DST_BRANCH" "$new_commit"
fi

git push "$REMOTE" "$DST_BRANCH:$DST_BRANCH"
echo "已推送 $(git rev-parse --short "$new_commit") 到 $REMOTE/$DST_BRANCH"
