# =======================================
# Author: Hung Tran-Nam
# Email: namhung34.info@gmail.com
# Repo: https://github.com/hungtrannam/probclust
# =======================================
# File: gitUp.sh
# Description: Script to initialize a Git repository and push to a remote repository


#!/bin/bash

REPO_URL="git@github.com:hungtrannam/probclust.git"
BRANCH="ver1"
DATETIME=$(date '+%Y-%m-%d %H:%M:%S')
MSG="[$DATETIME] Initial commit"

echo "🚀 Khởi tạo Git và đẩy lên $REPO_URL"

git init
git remote add origin "$REPO_URL"
git checkout -b "$BRANCH"
git add -A
git commit -m "$MSG"
git push -u origin "$BRANCH"
