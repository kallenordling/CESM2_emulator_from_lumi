#!/bin/bash
# Deploy (or update) the web app on CSC Rahti.
#
# Prerequisites, once:
#   1. Rahti enabled for a CSC project in https://my.csc.fi
#   2. `oc login https://api.2.rahti.csc.fi:6443 --token=...`
#      (console.rahti.csc.fi -> your name -> "Copy Login Command")
#   3. `hf auth login` done on this machine (the token is copied into a Rahti
#      secret so the app can read the private weights repo)
#
# Usage:
#   bash webapp/rahti/deploy.sh <rahti-namespace> <csc-project-number> [--github-webhook]
#
# Creates the namespace if needed, applies rahti.yaml, creates the secrets,
# builds the image from GitHub, waits for the rollout and prints the URL.
# --github-webhook also registers a GitHub push webhook so every push to the
# branch rebuilds and redeploys automatically (needs `gh` logged in).
set -euo pipefail

NS="${1:?usage: deploy.sh <rahti-namespace> <csc-project-number> [--github-webhook]}"
CSC_PROJECT="${2:?csc project number, e.g. 2001234}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GH_REPO="kallenordling/CESM2_emulator_from_lumi"

oc whoami >/dev/null || { echo "not logged in to Rahti: run oc login first" >&2; exit 1; }

if ! oc get project "${NS}" >/dev/null 2>&1; then
    oc new-project "${NS}" --description="csc_project: ${CSC_PROJECT}" \
        --display-name="CESM2 emission emulator"
fi
oc project "${NS}" >/dev/null

# secrets: created once, never printed
if ! oc get secret hf-token >/dev/null 2>&1; then
    HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HOME}/.cache/huggingface/token}"
    [[ -s "${HF_TOKEN_FILE}" ]] || { echo "no HF token at ${HF_TOKEN_FILE}" >&2; exit 1; }
    oc create secret generic hf-token --from-file=HF_TOKEN="${HF_TOKEN_FILE}" >/dev/null
    echo "created secret hf-token"
fi
if ! oc get secret github-webhook >/dev/null 2>&1; then
    oc create secret generic github-webhook \
        --from-literal=WebHookSecretKey="$(head -c 24 /dev/urandom | base64 | tr -dc A-Za-z0-9)" >/dev/null
    echo "created secret github-webhook"
fi

sed "s/__NAMESPACE__/${NS}/g" "${HERE}/rahti.yaml" | oc apply -f -

echo "building image on Rahti (first build ~10 min: torch + cartopy) ..."
oc start-build cesm2-emulator --follow --wait

oc rollout status deployment/cesm2-emulator --timeout=15m
HOST="$(oc get route cesm2-emulator -o jsonpath='{.spec.host}')"
echo
echo "App: https://${HOST}"

if [[ " $* " == *" --github-webhook "* ]]; then
    KEY="$(oc get secret github-webhook -o jsonpath='{.data.WebHookSecretKey}' | base64 -d)"
    API="$(oc whoami --show-server)"
    URL="${API}/apis/build.openshift.io/v1/namespaces/${NS}/buildconfigs/cesm2-emulator/webhooks/${KEY}/github"
    if gh api "repos/${GH_REPO}/hooks" --jq '.[].config.url' | grep -q "/namespaces/${NS}/buildconfigs/cesm2-emulator/"; then
        echo "GitHub webhook already registered"
    else
        gh api "repos/${GH_REPO}/hooks" -X POST \
            -f name=web -F active=true -f 'events[]=push' \
            -f config[url]="${URL}" -f config[content_type]=json >/dev/null
        echo "GitHub webhook registered: pushes to the BuildConfig branch now rebuild + redeploy"
    fi
fi
