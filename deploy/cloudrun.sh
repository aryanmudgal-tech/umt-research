#!/usr/bin/env bash
# Deploy the professor's web UI to Cloud Run. From the repository root:
#
#   PROJECT=your-project-id ./deploy/cloudrun.sh
#
# Optional: REGION (us-central1), SERVICE (beam-agent), BUCKET
# (<project>-beam-agent-runs), SECRET (gemini-api-key), DAILY_CAP (30),
# UPDATE_KEY=1 to push the key in .env as a new secret version, and
# BILLING_ACCOUNT=XXXXXX-XXXXXX-XXXXXX to add a $1 budget alert.
#
# Every step is safe to re-run. The Gemini key goes from .env straight into
# Secret Manager; it is never printed and never baked into the image.
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT to the Google Cloud project id}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-beam-agent}"
BUCKET="${BUCKET:-${PROJECT}-beam-agent-runs}"
SECRET="${SECRET:-gemini-api-key}"
DAILY_CAP="${DAILY_CAP:-30}"
G=(--project "$PROJECT" --quiet)

say() { printf '\n== %s\n' "$*"; }

say "Account and project"
gcloud auth list --filter=status:ACTIVE --format='value(account)'
echo "project: $PROJECT, region: $REGION, service: $SERVICE"

say "Enabling the APIs Cloud Run needs"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com storage.googleapis.com iam.googleapis.com \
  cloudresourcemanager.googleapis.com "${G[@]}"

say "Bucket for runs: gs://$BUCKET (a US region keeps it in the free tier)"
if ! gcloud storage buckets describe "gs://$BUCKET" "${G[@]}" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://$BUCKET" --location "$REGION" \
    --uniform-bucket-level-access --public-access-prevention "${G[@]}"
fi

say "Gemini key in Secret Manager: $SECRET"
if ! gcloud secrets describe "$SECRET" "${G[@]}" >/dev/null 2>&1; then
  gcloud secrets create "$SECRET" --replication-policy automatic "${G[@]}"
  UPDATE_KEY=1
fi
if [[ "${UPDATE_KEY:-0}" == "1" ]]; then
  key="$(grep -E '^GEMINI_API_KEY=' .env | head -1 | cut -d= -f2- | tr -d '"'"'"' \r')"
  [[ -n "$key" ]] || { echo "no GEMINI_API_KEY in .env" >&2; exit 1; }
  printf '%s' "$key" | gcloud secrets versions add "$SECRET" --data-file=- "${G[@]}" >/dev/null
  unset key
  echo "added a new version of $SECRET"
fi

say "Service accounts: one the site runs as, one that builds it"
# Dedicated accounts rather than the default compute one, which a project
# without Compute Engine does not have, and which can do far more than this.
ensure_sa() {
  if ! gcloud iam service-accounts describe "$1@${PROJECT}.iam.gserviceaccount.com" "${G[@]}" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$1" --display-name "$2" "${G[@]}"
  fi
}
ensure_sa "${SERVICE}-run" "Runs the $SERVICE web UI"
ensure_sa "${SERVICE}-build" "Builds the $SERVICE container"
runtime_sa="${SERVICE}-run@${PROJECT}.iam.gserviceaccount.com"
build_sa="${SERVICE}-build@${PROJECT}.iam.gserviceaccount.com"

# the site may read the key and read and write runs, nothing else
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member "serviceAccount:$runtime_sa" --role roles/storage.objectAdmin "${G[@]}" >/dev/null
gcloud secrets add-iam-policy-binding "$SECRET" \
  --member "serviceAccount:$runtime_sa" --role roles/secretmanager.secretAccessor "${G[@]}" >/dev/null
# the builder may run builds: read the uploaded source, push the image, write logs
gcloud projects add-iam-policy-binding "$PROJECT" --condition None \
  --member "serviceAccount:$build_sa" --role roles/cloudbuild.builds.builder "${G[@]}" >/dev/null

say "Building remotely and deploying (a few minutes the first time)"
# No login by decision, so the service is public; one instance, so one run at
# a time; 15-minute requests, because a run lives inside the open page.
gcloud run deploy "$SERVICE" --source . --region "$REGION" \
  --service-account "$runtime_sa" \
  --build-service-account "projects/${PROJECT}/serviceAccounts/${build_sa}" \
  --allow-unauthenticated --max-instances 1 --min-instances 0 --concurrency 20 \
  --memory 2Gi --cpu 1 --timeout 900 \
  --set-secrets "GEMINI_API_KEY=${SECRET}:latest" \
  --set-env-vars "RUNS_BUCKET=${BUCKET},DAILY_CAP=${DAILY_CAP}" "${G[@]}"

if [[ -n "${BILLING_ACCOUNT:-}" ]]; then
  say "Budget alert: email when spend passes \$1"
  gcloud services enable billingbudgets.googleapis.com "${G[@]}"
  gcloud billing budgets create --billing-account "$BILLING_ACCOUNT" \
    --display-name "$SERVICE budget" --budget-amount 1USD \
    --filter-projects "projects/$PROJECT" \
    --threshold-rule percent=0.5 --threshold-rule percent=1.0 --quiet >/dev/null
fi

say "Done"
gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)' "${G[@]}"
