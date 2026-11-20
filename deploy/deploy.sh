#!/usr/bin/env bash
# Server-side deployment of one commit of main, started by GitHub Actions over SSH (.github/workflows/deploy.yml).
#
# The deploy key in /root/.ssh/authorized_keys is pinned to this script (restrict,command="..."), so the only
# thing it can do is "deploy <40-hex sha>"; the request arrives in SSH_ORIGINAL_COMMAND. Steps:
#   1. check the request: the sha must be on origin/main and not older than what runs now
#   2. check out the commit (the app dir is a git checkout; .env, open-webui-data/, data indexes are ignored)
#   3. build all images first (a failed build changes nothing), then start what changed
#   4. health checks; on failure check out the previous commit again and restart the same way (rollback)
# Manual run as root on the server: SSH_ORIGINAL_COMMAND="deploy $(git rev-parse origin/main)" deploy/deploy.sh
set -Eeuo pipefail

APP_DIR=${APP_DIR:-/opt/open-webui}
LOG_FILE=${LOG_FILE:-/var/log/mjoelnir-deploy.log}
LOCK_FILE=${LOCK_FILE:-/run/lock/mjoelnir-deploy.lock}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-300}

log() {
	# Gibt eine Zeile mit Zeitstempel aus: im Actions-Log (stdout) und in LOG_FILE auf dem Server.
	printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG_FILE"
}

fail() {
	# Bricht mit einer Fehlermeldung ab; "::error::" wird in GitHub Actions als Fehler hervorgehoben.
	log "::error::$*"
	exit 1
}

changed() {
	# Prüft, ob sich zwischen zwei Commits eine Datei geändert hat, deren Pfad auf das Regex-Muster passt.
	git diff --name-only "$1" "$2" | grep -qE "$3"
}

wait_for() {
	# Wiederholt einen Prüfbefehl alle 5 s, bis er gelingt oder HEALTH_TIMEOUT abgelaufen ist.
	local name=$1 start=$SECONDS
	shift
	until "$@" >/dev/null 2>&1; do
		((SECONDS - start < HEALTH_TIMEOUT)) || { log "unhealthy: $name (no success within ${HEALTH_TIMEOUT}s)"; return 1; }
		sleep 5
	done
	log "healthy: $name ($((SECONDS - start))s)"
}

# shellcheck disable=SC2317  # called indirectly through wait_for
pipeline_loaded() {
	# Fragt den Pipelines-Server nach seinen Modellen; gesund nur, wenn die Capital-Pipeline geladen ist.
	# Der API-Key wird im Container aus dessen Umgebung gelesen und erscheint nirgends im Log.
	docker exec open-webui-pipelines-capital python -c '
import json, os, urllib.request
req = urllib.request.Request("http://localhost:9099/v1/models",
                             headers={"Authorization": "Bearer " + os.environ["PIPELINES_API_KEY"]})
ids = [m["id"] for m in json.load(urllib.request.urlopen(req, timeout=10))["data"]]
assert "capital_rag_pipeline" in ids, ids'
}

healthy() {
	# Alle Dienste prüfen: Open WebUI, Capital-Pipeline, Graph-Explorer und Caddy (Admin-Eingang auf localhost).
	wait_for "open-webui" curl -fsS http://127.0.0.1:3000/health &&
		wait_for "pipelines-capital" pipeline_loaded &&
		wait_for "graph-explorer" curl -fsS http://127.0.0.1:3001/ &&
		wait_for "caddy" curl -fsS http://127.0.0.1:8081/health
}

apply() {
	# Bringt die laufenden Container von Commit $1 auf Commit $2: auschecken, Images bauen, Dienste starten.
	# Einzeln eingebundene Dateien (Caddyfile, guest.css) ersetzt git durch eine neue Datei; der Container sieht
	# sie erst nach einem Neustart. Die Pipeline ist als Ordner eingebunden und wird nur beim Start geladen.
	# Called inside "if", where set -e does not apply: every step returns on failure explicitly.
	local from=$1 to=$2
	git checkout --quiet --detach "$to" || return 1
	log "checked out $(git log -1 --format='%h %s' "$to")"
	docker compose build --quiet || { log "image build failed"; return 1; }
	docker compose up -d --quiet-pull || { log "docker compose up failed"; return 1; }
	if changed "$from" "$to" '^(Caddyfile|guest\.css)$'; then
		docker compose run --rm --no-deps --entrypoint caddy caddy \
			validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null || { log "Caddyfile invalid"; return 1; }
		docker compose restart caddy || return 1
		log "restarted caddy (Caddyfile or guest.css changed)"
	fi
	if changed "$from" "$to" '^pipelines-capital/'; then
		docker compose restart pipelines-capital || return 1
		log "restarted pipelines-capital (pipeline code changed)"
	fi
}

main() {
	# Ablauf eines Deployments: Anfrage prüfen, sperren, ausrollen, prüfen, bei Fehler zurückrollen.
	local request=${SSH_ORIGINAL_COMMAND:-${1:-}}
	[[ $request =~ ^deploy\ ([0-9a-f]{40})$ ]] || fail "expected 'deploy <40-hex sha>', got '${request:0:80}'"
	local sha=${BASH_REMATCH[1]}

	exec 9>"$LOCK_FILE"
	flock -w 600 9 || fail "another deployment is still running"
	cd "$APP_DIR"

	git fetch --quiet origin main
	git merge-base --is-ancestor "$sha" origin/main || fail "$sha is not on origin/main"
	local prev
	prev=$(git rev-parse HEAD)
	if [[ $sha == "$prev" ]]; then
		log "$sha is already deployed - health check only"
		healthy || fail "services unhealthy"
		return
	fi
	# Older commits are not deployed automatically (a late CI run must not roll the server back).
	if git merge-base --is-ancestor "$sha" "$prev"; then
		fail "$sha is older than the deployed $prev"
	fi
	# Hand edits on the server would be overwritten: stop and let a human decide. valves.json is runtime state
	# (the pipelines server writes it when valves are changed in Open WebUI); git keeps it across the checkout.
	local edits
	edits=$(git status --short --untracked-files=no -- . ':!pipelines-capital/*/valves.json')
	[[ -z $edits ]] || fail "local changes on the server: $(tr '\n' ' ' <<<"$edits")"

	log "deploying $sha (was $prev)"
	if apply "$prev" "$sha" && healthy; then
		log "deployed $sha"
		return
	fi
	log "::warning::deployment of $sha failed - rolling back to $prev"
	if apply "$sha" "$prev" && healthy; then
		fail "deployment of $sha failed; rolled back to $prev"
	fi
	fail "deployment of $sha failed and the rollback to $prev is unhealthy too - check the server"
}

# An SSH disconnect must not stop a deployment halfway. main runs as a whole function, so a git checkout that
# replaces this file while it runs does not change the code being executed.
trap '' HUP
main "$@"
exit
