#!/bin/bash
# 🚀 BasicChat Release Management Script
# Usage: ./scripts/release.sh [patch|minor|major|rc|promote <rc-version>]
set -e
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }
check_branch() { current_branch=$(git branch --show-current); if [ "$current_branch" != "main" ]; then log_error "Must be on main branch to release. Current branch: $current_branch"; exit 1; fi }
check_clean() { if [ -n "$(git status --porcelain)" ]; then log_error "Working directory is not clean. Please commit or stash changes."; git status --short; exit 1; fi }
get_current_version() { grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'; }
bump_version() { local bump_type=$1; local current_version=$(get_current_version); current_version=${current_version#v}; IFS='.' read -ra VERSION_PARTS <<< "$current_version"; major=${VERSION_PARTS[0]}; minor=${VERSION_PARTS[1]}; patch=${VERSION_PARTS[2]}; case $bump_type in "patch") new_patch=$((patch + 1)); new_version="$major.$minor.$new_patch";; "minor") new_minor=$((minor + 1)); new_version="$major.$new_minor.0";; "major") new_major=$((major + 1)); new_version="$new_major.0.0";; "rc") new_patch=$((patch + 1)); new_version="$major.$minor.$new_patch-rc.1";; *) log_error "Invalid bump type: $bump_type. Use: patch, minor, major, or rc"; exit 1;; esac; echo "v$new_version"; }
update_version() { local version=$1; local version_without_v=${version#v}; log_info "Updating version to $version in files..."; sed -i.bak "s/^version = \".*\"/version = \"$version_without_v\"/" pyproject.toml; rm pyproject.toml.bak; if [ -f package.json ]; then sed -i.bak "s/\"version\": \".*\"/\"version\": \"$version_without_v\"/" package.json; rm package.json.bak; fi; log_success "Version updated to $version"; }
run_checks() { log_info "Running pre-release checks..."; log_info "Running test suite..."; python -m pytest -n auto tests/ -v --tb=short; log_info "Running E2E tests..."; bunx playwright test --reporter=list; log_info "Running performance tests..."; python scripts/test_performance_regression.py; log_success "All checks passed!"; }
create_release() { local version=$1; local is_rc=$2; log_info "Creating release $version..."; git add pyproject.toml package.json; git commit -m "chore: bump version to $version"; git tag -a "$version" -m "Release $version"; git push origin main; git push origin "$version"; if [ "$is_rc" = "true" ]; then log_success "Release candidate $version created and pushed!"; log_warning "To promote to production, run: ./scripts/release.sh promote $version"; else log_success "Production release $version created and pushed!"; fi }
promote_rc() { local rc_version=$1; if [[ ! $rc_version =~ ^v[0-9]+\.[0-9]+\.[0-9]+-rc\.[0-9]+$ ]]; then log_error "Invalid RC version format: $rc_version"; exit 1; fi; prod_version=${rc_version%-rc.*}; log_info "Promoting $rc_version to $prod_version..."; git tag -a "$prod_version" -m "Production release $prod_version" "$rc_version"; git push origin "$prod_version"; log_success "Promoted $rc_version to production release $prod_version!"; }
main() { local action=$1; local version=$2; case $action in "patch"|"minor"|"major"|"rc") check_branch; check_clean; run_checks; new_version=$(bump_version "$action"); update_version "$new_version"; is_rc=false; if [ "$action" = "rc" ]; then is_rc=true; fi; create_release "$new_version" "$is_rc";; "promote") if [ -z "$version" ]; then log_error "Please provide RC version to promote"; exit 1; fi; promote_rc "$version";; *) echo "Usage: $0 [patch|minor|major|rc|promote <rc-version>]"; echo ""; echo "Commands:"; echo "  patch    - Bump patch version (1.0.0 -> 1.0.1)"; echo "  minor    - Bump minor version (1.0.0 -> 1.1.0)"; echo "  major    - Bump major version (1.0.0 -> 2.0.0)"; echo "  rc       - Create release candidate (1.0.0 -> 1.0.1-rc.1)"; echo "  promote  - Promote RC to production"; echo ""; echo "Examples:"; echo "  $0 patch"; echo "  $0 rc"; echo "  $0 promote v1.0.1-rc.1"; exit 1;; esac }
main "$@" 