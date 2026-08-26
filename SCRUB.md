# Public scrub evidence

## Scope

The scrub covers every file in the Git index. It reads each file as bytes, rejects symlinks, and checks these named categories:

- hostnames
- home paths
- internal and tailnet IP addresses
- email addresses
- private keys, bearer credentials, API keys, access credentials, passwords, and secrets
- SSH aliases and user-qualified SSH targets
- tailnet names and internal hub domains

## Commands

Run the complete check:

```bash
python3 scripts/scrub.py
```

Review the exact high-risk strings independently:

```bash
git grep -nI -E 'chris[-](5090|ms|macbook|laptop)|/(home|Users)/[c]hris|100([.]([0-9]{1,3})){3}|[.]ts[.]net|egd[a]tahub|B[e]arer[[:space:]]+[A-Za-z0-9._~+/=-]{12,}|BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY|ssh[[:space:]]+5090|chris[@]' -- . ':!SCRUB.md'
git grep -nI -E '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+[.][A-Za-z]{2,}' -- . ':!SCRUB.md'
```

Both `git grep` commands must exit 1 with no output. Exit 1 means no match.

## Result

Final run after staging the complete repository:

```text
checked 70 indexed files, 44672708 bytes
hostnames: PASS
home paths: PASS
internal IPs: PASS
emails: PASS
tokens: PASS
ssh aliases: PASS
tailnet names: PASS
symlinks: PASS
scrub verdict: PASS
```

Verdict: **PASS**. No listed category matched any public file. The independent grep commands returned no output. No symlink is present.

This evidence applies to the initial public commit only. Run the commands again before every push.
