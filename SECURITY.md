# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | ✅                 |
| < 1.0    | ❌                 |

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub issues.

Instead, email [myneluca@gmail.com](mailto:myneluca@gmail.com) with a detailed description of the issue.

You should receive a response within 48 hours. If the issue is confirmed, we will release a patch as soon as possible, typically within one week.

## Scope

Security-relevant areas of TinyBPE include:

- **C extension memory safety**: The `src/` directory contains pure C code that handles arbitrary data. Buffer overflows, use-after-free, and memory leaks in the C tokenizer or trainer are considered security issues.
- **Model file parsing**: The `.tbm` and `.vocab` file readers parse untrusted input. Integer overflow or memory exhaustion when loading malicious model files is in scope.
- **Denial of service**: Pathological inputs that cause excessive CPU or memory usage during encoding/decoding.

## Out of Scope

- Issues that require local access to modify installed package files
- Issues in conversion scripts (`scripts/`) which are not part of the installed package
- Theoretical attacks with no practical exploit vector
