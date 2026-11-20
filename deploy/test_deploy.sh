#!/usr/bin/env bash
# Tests for deploy/deploy.sh without a server: a local git "origin" with main, an app checkout, and stubs for
# docker and curl that log their calls. Covers request validation, deploy, no-op, refusal of older commits and
# local edits, and the automatic rollback when a health check fails. Needs only bash, git and flock.
#   bash deploy/test_deploy.sh        (CI runs it on every push)
set -u

DEPLOY=$(cd "$(dirname "$0")" && pwd)/deploy.sh
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
export GIT_CONFIG_GLOBAL=$WORK/gitconfig
git config --global user.email test@example.com
git config --global user.name test
git config --global init.defaultBranch main
git config --global advice.detachedHead false

# Stubs: docker fails the pipeline health check while the commit in FAIL_HEALTH is checked out.
mkdir -p "$WORK/stub"
cat >"$WORK/stub/docker" <<'EOF'
#!/usr/bin/env bash
echo "docker $*" >>"$CALLS"
[[ "$*" == "compose up"* ]] && echo "APP_VERSION=${APP_VERSION:-}" >>"$CALLS"
if [[ "$*" == *"exec open-webui-pipelines-capital"* && -n "${FAIL_HEALTH:-}" ]] \
	&& [[ "$(git -C "$APP_DIR" rev-parse HEAD)" == "$FAIL_HEALTH" ]]; then
	exit 1
fi
EOF
cat >"$WORK/stub/curl" <<'EOF'
#!/usr/bin/env bash
echo "curl $*" >>"$CALLS"
EOF
chmod +x "$WORK/stub/docker" "$WORK/stub/curl"
export PATH=$WORK/stub:$PATH APP_DIR=$WORK/app LOG_FILE=$WORK/deploy.log LOCK_FILE=$WORK/deploy.lock
export CALLS=$WORK/calls HEALTH_TIMEOUT=2

# Origin with three commits on main (c2 changes the pipeline, c3 the Caddyfile) and one on another branch.
git init -q --bare "$WORK/origin.git"
git clone -q "$WORK/origin.git" "$WORK/dev" 2>/dev/null
cd "$WORK/dev" || exit 1
commit() { echo "$2" >"$1" && git add "$1" && git commit -qm "$3" && git push -q origin HEAD && git rev-parse HEAD; }
mkdir -p pipelines-capital/capital_rag_pipeline
commit pipelines-capital/capital_rag_pipeline/valves.json '{}' c0 >/dev/null
C1=$(commit Caddyfile a c1)
C2=$(commit pipelines-capital/p.py p2 c2)
C3=$(commit Caddyfile b c3)
git tag v1.0.0 "$C2" && git push -q origin v1.0.0
git checkout -qb other
OTHER=$(commit x x other)
git clone -q "$WORK/origin.git" "$APP_DIR"
git -C "$APP_DIR" checkout -q --detach "$C1"

passed=0 failed=0
check() {
	# Führt einen Befehl aus und vergleicht seinen Exit-Code mit dem erwarteten; vor Deploy-Aufrufen wird das
	# Aufrufprotokoll der Stubs geleert, damit die folgenden grep-Prüfungen nur diesen Lauf sehen.
	local name=$1 want=$2
	shift 2
	[[ $1 == deploy || $1 == env ]] && : >"$CALLS"
	local got=0
	"$@" >"$WORK/out" 2>&1 || got=$?
	if [[ $got == "$want" ]]; then
		passed=$((passed + 1))
		echo "ok   $name"
	else
		failed=$((failed + 1))
		echo "FAIL $name (exit $got, expected $want)"
		sed 's/^/     /' "$WORK/out"
	fi
}
deploy() { SSH_ORIGINAL_COMMAND="deploy $1" bash "$DEPLOY"; }
head_is() { [[ $(git -C "$APP_DIR" rev-parse HEAD) == "$1" ]]; }
called() { grep -q "$1" "$CALLS"; }

check "rejects a request other than 'deploy <sha>'" 1 env SSH_ORIGINAL_COMMAND="rm -rf /" bash "$DEPLOY"
check "rejects a commit that is not on main" 1 deploy "$OTHER"
check "deploys c2" 0 deploy "$C2"
check "  c2 is checked out" 0 head_is "$C2"
check "  pipeline changed: pipelines-capital restarted" 0 called "restart pipelines-capital"
check "  Caddyfile unchanged: caddy not restarted" 1 called "restart caddy"
check "  release tag passed on as APP_VERSION" 0 called "APP_VERSION=v1.0.0$"
check "same commit again: health check only" 0 deploy "$C2"
check "refuses the older c1" 1 deploy "$C1"
check "  c2 still checked out" 0 head_is "$C2"
check "c3 fails its health check" 1 env FAIL_HEALTH="$C3" SSH_ORIGINAL_COMMAND="deploy $C3" bash "$DEPLOY"
check "  rolled back to c2" 0 head_is "$C2"
check "  Caddyfile validated before the restart" 0 called "validate --config"
check "  rollback in the log" 0 grep -q "rolled back to $C2" "$LOG_FILE"
echo edit >>"$APP_DIR/Caddyfile"
check "refuses to overwrite local edits" 1 deploy "$C3"
git -C "$APP_DIR" checkout -q -- Caddyfile
# Valves changed in Open WebUI: the pipelines server rewrote valves.json; that must not block a deployment.
VALVES=$APP_DIR/pipelines-capital/capital_rag_pipeline/valves.json
echo '{"SHOW_COSTS": false}' >"$VALVES"
check "deploys c3 although valves.json was changed at runtime" 0 deploy "$C3"
check "  Caddyfile changed: caddy restarted" 0 called "restart caddy"
check "  c3 is checked out" 0 head_is "$C3"
check "  runtime valves kept" 0 grep -q SHOW_COSTS "$VALVES"
# A large diff (more than the 64 KB pipe buffer after the matching path) must still detect the pipeline change.
git checkout -q main
mkdir -p zz && for i in $(seq 6000); do echo "$i" >"zz/file-$i.txt"; done
echo p4 >pipelines-capital/p.py && git add -A && git commit -qm c4 && git push -q origin HEAD:main
C4=$(git rev-parse HEAD)
check "deploys c4 with a large diff" 0 deploy "$C4"
check "  pipeline change detected in the large diff" 0 called "restart pipelines-capital"

echo "$passed passed, $failed failed"
[[ $failed == 0 ]]
