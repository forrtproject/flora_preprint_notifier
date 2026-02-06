# Agent Notes

## PR Body Quoting (Critical)

When creating or editing PR descriptions with GitHub CLI, never pass Markdown containing backticks using an inline double-quoted `--body` string. Shell command substitution will corrupt the text.

Use one of these safe patterns:

1. Preferred: write to a file, then pass `--body-file`.
```bash
cat > /tmp/pr_body.md <<'EOF'
## Summary
- keep code spans like `docker-compose.yml` and `GROBID_URL` intact
EOF
gh pr create --title "..." --base main --head <branch> --body-file /tmp/pr_body.md
```

2. For updates:
```bash
gh pr edit <number> --body-file /tmp/pr_body.md
```

Do not use:
```bash
gh pr create --body "text with `backticks`"
```
