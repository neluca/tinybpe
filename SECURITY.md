# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

TinyBPE takes security seriously. If you discover a security vulnerability,
please **do not** open a public issue.

Instead, please report it privately via email to: **myneluca@gmail.com**

Please include the following in your report:
- A description of the vulnerability
- Steps to reproduce the issue
- Affected versions
- Any potential mitigations you've identified

You can expect:
- An acknowledgment of your report within 48 hours
- A timeline for resolution within 7 days
- Credit in the release notes (unless you prefer to remain anonymous)

## Scope

Security issues in scope include:
- Memory safety issues (buffer overflows, use-after-free, null pointer dereference)
- Integer overflow leading to incorrect behavior
- Denial of service via crafted input
- Issues that could lead to arbitrary code execution

## Out of Scope

- Denial of service via resource exhaustion with legitimate input sizes
- Issues in example scripts that are not part of the library itself
