#!/usr/bin/env bash
# Upload a finished run to s3://molcensus-results-<account id>/<run stamp>/ and check that every file
# arrived. run_all.sh calls it after the last stage and stops the instance at once only if it exits 0.
# Everything it does goes to ~/runs/upload.log; a failure ends with "UPLOAD FAILED: <why>".
#   bash ~/molcensus/scripts/aws/upload_results.sh <run stamp>
#
# The folders in the bucket are the ones "Copy back from S3" (scripts/aws/README.md) copies to the Mac:
#   results/aws*/                            from ~/molcensus/results/aws*
#   outputs/work/biominer_output/            from ~/molcensus/outputs/work/biominer_output
#   outputs/work_biovista/biominer_output/   from ~/molcensus/outputs/work_biovista/biominer_output
#   runs/ and setup.log                      from ~/runs and ~/setup.log
# Left out: PDFs, because MinerU keeps a copy of every paper BioMiner reads (the course PDFs among
# them), and runs/upload.log, which is still being written.
# The check: the run's folder in the bucket holds as many files as were sent. Only then this script
# writes UPLOAD_COMPLETE there (file count and time).
set -uo pipefail
MOLCENSUS=$(cd "$(dirname "$0")/../.." && pwd)
STAMP=${1:?usage: upload_results.sh <run stamp>}
export AWS_DEFAULT_REGION=us-east-2  # the bucket's region
mkdir -p "$HOME/runs"
exec >> "$HOME/runs/upload.log" 2>&1
echo "=== upload started $(date -u +%FT%TZ)"

fail() { echo "UPLOAD FAILED: $*"; exit 1; }

ACCOUNT=$(aws sts get-caller-identity --query Account --output text) \
  || fail "no AWS credentials; is the molcensus-ec2 instance profile attached? (scripts/aws/README.md)"
DEST=s3://molcensus-results-$ACCOUNT/$STAMP
sent=0

send() {  # send <local folder or file> <its path under $DEST>
  if [ -d "$1" ]; then
    # In aws filters '*' also matches '/', so '*.pdf' leaves out PDFs at any depth, as find does below.
    # 'upload.log' only matches at the top of the folder, as -path does.
    aws s3 sync "$1" "$DEST/$2" --only-show-errors --exclude '*.pdf' --exclude '*.PDF' --exclude upload.log \
      || fail "aws s3 sync $1 failed (exit $?)"
    sent=$(( sent + $(find -L "$1" -type f ! -name '*.pdf' ! -name '*.PDF' ! -path "$1/upload.log" | grep -c .) ))
    echo "sent $1 to $DEST/$2/"
  elif [ -f "$1" ]; then
    aws s3 cp "$1" "$DEST/$2" --only-show-errors || fail "aws s3 cp $1 failed (exit $?)"
    sent=$(( sent + 1 ))
    echo "sent $1 to $DEST/$2"
  else
    echo "not there, skipped: $1"
  fi
}

cd "$MOLCENSUS" || fail "no $MOLCENSUS"
for folder in results/aws*; do send "$folder" "$folder"; done
send outputs/work/biominer_output outputs/work/biominer_output
send outputs/work_biovista/biominer_output outputs/work_biovista/biominer_output
send "$HOME/runs" runs
send "$HOME/setup.log" setup.log

listing=$(aws s3 ls --recursive "$DEST/") || fail "cannot list $DEST/"
in_bucket=$(printf '%s\n' "$listing" | grep -c .)
[ "$in_bucket" -eq "$sent" ] || fail "$sent files sent but $in_bucket in $DEST/"
echo "$sent files, checked $(date -u +%FT%TZ)" | aws s3 cp - "$DEST/UPLOAD_COMPLETE" --only-show-errors \
  || fail "could not write $DEST/UPLOAD_COMPLETE"
echo "UPLOAD_COMPLETE: $sent files in $DEST/"
